#!/bin/bash
#
# One submitter for the retrievals in this directory.
#
#   ./submit.sh <nodes> <ranks-per-node> <resolution-k> [branch] [solver]
#
#   nodes             nodes to allocate; 1 for jax unless SMC is sharded
#   ranks-per-node    MPI ranks per node, or the cores of the one jax task
#   resolution-k      cross-section resolution in k: 1, 15, 50, 100
#   branch            jax (taurex4, .venv_jax, default) | original (taurex3, .venv)
#   solver            multinest [default] | jax | laplace | nuts | mclmc | smc
#
# solver=jax is the differentiability plugin's optimizer: it finds the mode from
# the model gradient and draws with JAX_SAMPLER (laplace, nuts, mclmc, smc; smc
# also returns log Z). laplace/nuts/mclmc/smc are shorthand for jax + sampler.
# smc shards its particle axis over the MPI ranks, so <nodes> x <ranks-per-node>
# are separate processes and each takes a contiguous shard; SMC_MPI=0 runs it
# as one task on one node with XLA threads instead. The other samplers and the
# mode fit are always one process on one node, where <ranks-per-node> is its
# core count: extra ranks would only duplicate the compiled model.
#
# It reads parfile.par and ../Inputs, writes the run's parfile (cross-sections,
# chain path and the solver's [Optimizer] section) and submits a script kept in
# $SCRIPT_DIR. The run tag carries the solver, so samplers do not overwrite.
#
#   ./submit.sh 2 64 50                    # MultiNest, 2x64 ranks
#   ./submit.sh 4 32 50 original           # MultiNest, taurex3, 4x32 ranks
#   ./submit.sh 1 128 50 jax laplace       # jax laplace, 1 task on 128 cores
#   ./submit.sh 2 48 50 jax smc            # SMC sharded over 2x48 ranks
#   SMC_MPI=0 ./submit.sh 1 128 50 jax smc # SMC as one task on 128 cores
#
# MultiNest reserves the whole node (--ntasks-per-node) and places
# <ranks-per-node> on it with --map-by ppr; without ppr every rank lands on one
# node and the job OOMs. See parallelism_laplace.md for the jax layout.
#
# Environment overrides:
#   SMC_MPI=1               1 shards SMC's particles over the MPI ranks, 0 runs
#                           it as one task on one node
#   JAX_SAMPLER=laplace     laplace | nuts | mclmc | smc
#   JAX_SMC_TARGET_ESS=0.5  smc: ESS fraction kept at the end of a stage
#   JAX_SMC_POLISH_ESS=0.15 smc: proposal ESS above which one beta=1 pass
#                           replaces the tempering ladder
#   JAX_SMC_MAX_DELTA=1.0   smc: largest tempering step; 1.0 lets the ESS
#                           schedule reach beta=1 in one stage when it can
#   JAX_SMC_MIN_DELTA=0.0   smc: smallest tempering step; >0 caps stages at 1/min
#   JAX_SMC_MAX_STEPS=50    smc: maximum number of stages
#   JAX_NUM_MCMC_STEPS=1    smc: HMC transitions per particle per stage
#   JAX_SMC_HMC_STEPS=3     smc: leapfrogs per transition
#   JAX_NUM_PARTICLES=128   smc: particles in the single-process layout; the
#                           sharded layout keeps two per rank for memory
#   JAX_SMC_START=laplace   smc: laplace (mode proposal) | prior (ladder from 0)
#   JAX_SMC_STEP_SIZE=0.5   smc: initial HMC step size
#   JAX_SMC_BATCH_SIZE=     smc: particles per batch; unset sizes it from 4 GiB
#   JAX_NUM_SAMPLES=2000    nuts and mclmc draws, split over JAX_NUM_CHAINS=16
#   JAX_NUM_WARMUP=150      nuts: adaptation steps per chain
#   JAX_PILOT_SAMPLES=0     nuts/smc: NUTS pilot draws blended into the metric
#   JAX_PILOT_WARMUP=75     nuts/smc: warm-up steps of that pilot
#   JAX_PILOT_BLEND=        nuts/smc: Laplace weight in the blend (default auto)
#   JAX_MAX_ITERATIONS=200  L-BFGS steps of the mode fit; JAX_N_STARTS=1 starts
#   JAX_CACHE_DIR=$PWD/.jax_cache   exported as JAX_COMPILATION_CACHE_DIR; the
#                           ranks share compiled programs through it
#   XLA_OPT_LEVEL=        if set, XLA_FLAGS=--xla_backend_optimization_level=N;
#                           a lower N spends less time compiling the model graph
#   LIGHT=1 EXCLUSIVE=1 PARTITION=rome TIME=03:00:00 SCRIPT_DIR=/tmp/taurex_submit

usage() {
    echo "usage: $0 <nodes> <ranks-per-node> <resolution-k> [jax|original] [multinest|jax|laplace|nuts|mclmc|smc]" >&2
    exit 2
}

[ $# -ge 3 ] || usage
NODES=$1 RANKS_PER_NODE=$2 RES_K=$3 BRANCH=${4:-jax} SOLVER=${5:-multinest}

case "$NODES$RANKS_PER_NODE" in ''|*[!0-9]*) usage;; esac
[ "$NODES" -ge 1 ] && [ "$RANKS_PER_NODE" -ge 1 ] || usage

case "$RES_K" in
    1|15|100) XSEC_PATH=../Inputs/xsec_R${RES_K}k ;;
    50)       XSEC_PATH=../Inputs/xsec_exomol_R50k ;;
    *) echo "unknown resolution k: $RES_K (known: 1, 15, 50, 100)" >&2; exit 2 ;;
esac

case "$BRANCH" in
    jax)      VENV=.venv_jax ;;
    original) VENV=.venv ;;
    *) echo "unknown branch: $BRANCH (known: jax, original)" >&2; exit 2 ;;
esac

# laplace|nuts|mclmc|smc are shorthand for solver=jax with that sampler
case "$SOLVER" in
    laplace|nuts|mclmc|smc) JAX_SAMPLER=$SOLVER; SOLVER=jax ;;
esac
case "$SOLVER" in
    multinest|jax) ;;
    *) echo "unknown solver: $SOLVER (known: multinest, jax, laplace, nuts, mclmc, smc)" >&2; exit 2 ;;
esac
JAX_SAMPLER=${JAX_SAMPLER:-laplace}
case "$JAX_SAMPLER" in
    laplace|nuts|mclmc|smc) ;;
    *) echo "unknown sampler: $JAX_SAMPLER (known: laplace, nuts, mclmc, smc)" >&2; exit 2 ;;
esac
[ "$SOLVER" = jax ] && [ "$BRANCH" != jax ] && { echo "solver jax needs branch jax" >&2; exit 2; }

RANKS_TOTAL=$((NODES * RANKS_PER_NODE))
# 0 keeps SMC in one process on one node with XLA threading: one compile of the
# model graph and one copy of the model, which is what the wall clock is spent
# on. 1 shards the particle axis over the ranks and compiles once behind a
# barrier, which only pays off when the persistent cache is shared
SMC_MPI=${SMC_MPI:-0}
MPI_SMC=0
if [ "$SOLVER" = jax ] && [ "$JAX_SAMPLER" = smc ] && [ "$SMC_MPI" = 1 ]; then MPI_SMC=1; fi

JAX_NUM_SAMPLES=${JAX_NUM_SAMPLES:-2000}
JAX_MAX_ITERATIONS=${JAX_MAX_ITERATIONS:-200}
JAX_N_STARTS=${JAX_N_STARTS:-1}
JAX_NUM_WARMUP=${JAX_NUM_WARMUP:-150}
JAX_NUM_CHAINS=${JAX_NUM_CHAINS:-16}
JAX_PILOT_WARMUP=${JAX_PILOT_WARMUP:-75}
JAX_PILOT_SAMPLES=${JAX_PILOT_SAMPLES:-0}
JAX_PILOT_BLEND=${JAX_PILOT_BLEND:-}
# two particles per rank: one leaves the collectives to dominate, more only
# lengthens every stage. Fixed here, not an option, because the rank count
# already decides how many particles a run has.
JAX_NUM_MCMC_STEPS=${JAX_NUM_MCMC_STEPS:-1}
JAX_SMC_HMC_STEPS=${JAX_SMC_HMC_STEPS:-2}
# none: no mode search at all, so no gradient of the model and no four-minute
# compilation of a fit program; the run tempers from the prior instead. adam
# fits the mode with plain Adam steps, lbfgs with the quasi-Newton fit and its
# own inverse Hessian as the proposal.
JAX_SMC_FIT=${JAX_SMC_FIT:-none}
JAX_SMC_START=${JAX_SMC_START:-$([ "$JAX_SMC_FIT" = none ] && echo prior || echo laplace)}
# auto: the L-BFGS inverse Hessian of a converged fit, else the Laplace Hessian
JAX_SMC_METRIC=${JAX_SMC_METRIC:-auto}
# steps of the Adam mode fit: it runs on one core while the ranks wait, so this
# is a serial cost that only buys a better proposal
JAX_SMC_FIT_STEPS=${JAX_SMC_FIT_STEPS:-60}
# the ESS fraction kept per stage: the number of stages is set by it, roughly
# KL / log(1/fraction), so a smaller value means fewer, larger stages at a lower
# ESS each, which the HMC moves make up for
JAX_SMC_TARGET_ESS=${JAX_SMC_TARGET_ESS:-0.25}
JAX_SMC_POLISH_ESS=${JAX_SMC_POLISH_ESS:-0.15}
JAX_SMC_MAX_DELTA=${JAX_SMC_MAX_DELTA:-1.0}
JAX_SMC_MIN_DELTA=${JAX_SMC_MIN_DELTA:-0.0}
# a stage costs two particles per rank, so the cap is set where a ladder from
# the prior can reach beta = 1 instead of stopping on a tempered target
JAX_SMC_MAX_STEPS=${JAX_SMC_MAX_STEPS:-200}
JAX_SMC_STEP_SIZE=${JAX_SMC_STEP_SIZE:-0.5}
JAX_SMC_BATCH_SIZE=${JAX_SMC_BATCH_SIZE:-}

if [ "$MPI_SMC" = 1 ]; then
    # two particles per rank: one leaves the collectives to dominate, more only
    # lengthens every stage. Fixed here because the rank count decides the
    # memory a sharded run needs (each rank holds a compiled model as well).
    JAX_NUM_PARTICLES=$((RANKS_TOTAL * 2))
else
    # one process over the whole node; more particles are linear work and buy
    # posterior resolution, so this is the knob for a final run
    JAX_NUM_PARTICLES=${JAX_NUM_PARTICLES:-128}
    # Fastest layout: evaluate the whole population in one vmapped batch so
    # XLA sees every particle at once, instead of the 4 GiB internal budget's
    # ~15. Cap at 128 so a larger particle count stays inside the node memory
    # (128 x ~220 MB at R50k/80 layers is about 28 GB).
    if [ -z "$JAX_SMC_BATCH_SIZE" ]; then
        JAX_SMC_BATCH_SIZE=$JAX_NUM_PARTICLES
        [ "$JAX_SMC_BATCH_SIZE" -gt 128 ] && JAX_SMC_BATCH_SIZE=128
    fi
fi

# smc_min_delta > max_delta only fails in the optimizer, after the mode fit
if [ "$SOLVER" = jax ] && [ "$JAX_SAMPLER" = smc ]; then
    awk -v lo="$JAX_SMC_MIN_DELTA" -v hi="$JAX_SMC_MAX_DELTA" 'BEGIN { exit !(lo >= 0 && lo <= hi) }' \
        || { echo "smc needs 0 <= JAX_SMC_MIN_DELTA <= JAX_SMC_MAX_DELTA" >&2; exit 2; }
fi

SOLVER_TAG=$SOLVER
JAX_BLOCK=
if [ "$SOLVER" = jax ]; then
    SOLVER_TAG=jax_${JAX_SAMPLER}
    # these keys go straight to the optimizer constructor, so a sampler's keys
    # are written only for that sampler: nuts takes num_chains and num_warmup,
    # mclmc only chains, smc particles and a step size. The remaining options
    # (seed, device, jitter, ...) keep their defaults.
    JAX_BLOCK="optimizer = jax
sampler = ${JAX_SAMPLER}
num_samples = ${JAX_NUM_SAMPLES}
max_iterations = ${JAX_MAX_ITERATIONS}
n_starts = ${JAX_N_STARTS}"
    case "$JAX_SAMPLER" in
        nuts) JAX_BLOCK="${JAX_BLOCK}
num_warmup = ${JAX_NUM_WARMUP}
num_chains = ${JAX_NUM_CHAINS}" ;;
        mclmc) JAX_BLOCK="${JAX_BLOCK}
num_chains = ${JAX_NUM_CHAINS}" ;;
        smc) JAX_BLOCK="${JAX_BLOCK}
num_particles = ${JAX_NUM_PARTICLES}
num_mcmc_steps = ${JAX_NUM_MCMC_STEPS}
smc_hmc_steps = ${JAX_SMC_HMC_STEPS}
smc_start = ${JAX_SMC_START}
smc_metric = ${JAX_SMC_METRIC}
smc_fit = ${JAX_SMC_FIT}
smc_target_ess = ${JAX_SMC_TARGET_ESS}
smc_polish_ess = ${JAX_SMC_POLISH_ESS}
smc_max_delta = ${JAX_SMC_MAX_DELTA}
smc_min_delta = ${JAX_SMC_MIN_DELTA}
smc_max_steps = ${JAX_SMC_MAX_STEPS}
smc_max_iterations = ${JAX_SMC_FIT_STEPS}
smc_step_size = ${JAX_SMC_STEP_SIZE}"
            # unset: the optimizer sizes the batch from its 4 GiB budget
            [ -n "$JAX_SMC_BATCH_SIZE" ] && JAX_BLOCK="${JAX_BLOCK}
smc_batch_size = ${JAX_SMC_BATCH_SIZE}" ;;
    esac
    # The pilot is a short NUTS run from the mode whose covariance is blended
    # with the Laplace one; it matters only for nuts and smc.
    if [ "$JAX_PILOT_SAMPLES" -gt 0 ]; then
        case "$JAX_SAMPLER" in
            nuts|smc) JAX_BLOCK="${JAX_BLOCK}
pilot_warmup = ${JAX_PILOT_WARMUP}
pilot_samples = ${JAX_PILOT_SAMPLES}"
                [ -n "$JAX_PILOT_BLEND" ] && JAX_BLOCK="${JAX_BLOCK}
pilot_blend = ${JAX_PILOT_BLEND}"
                ;;
        esac
    fi
    JAX_BLOCK="${JAX_BLOCK}
"
fi

# the plugin and the sampler's library are needed after the mode fit, so a
# missing one would only show up at the end of a long job
if [ "$SOLVER" = jax ]; then
    [ -x "$VENV/bin/python" ] || { echo "$VENV/bin/python not found" >&2; exit 2; }
    MODS=taurex.differentiability
    case "$JAX_SAMPLER" in
        smc) MODS="$MODS blackjax.smc.tempered" ;;
        laplace) ;;
        *) MODS="$MODS blackjax" ;;
    esac
    [ "$MPI_SMC" = 1 ] && MODS="$MODS mpi4py"
    for m in $MODS; do
        "$VENV/bin/python" -c "import $m" >/dev/null || { echo "$VENV cannot import $m: install the jax deps" >&2; exit 2; }
    done
fi

# cores per node, and the Slurm wall time
NTASKS_PER_NODE=128
TIME=${TIME:-03:00:00}
[ "$RANKS_PER_NODE" -le "$NTASKS_PER_NODE" ] || {
    echo "$RANKS_PER_NODE ranks/node does not fit in a node of $NTASKS_PER_NODE cores" >&2
    exit 2
}

# ---------------- launch layout (parallelism_laplace.md) ----------------
# solver=jax with SMC_MPI=0 is one task on one node, with <ranks-per-node>
# read as its core count: XLA spreads every compiled program over those cores.
if [ "$SOLVER" = jax ] && [ "$MPI_SMC" = 0 ]; then
    [ "$NODES" -eq 1 ] || {
        echo "solver jax runs the fit in one process on one node: pass 1 as <nodes>" >&2
        exit 2
    }
    CORES=$RANKS_PER_NODE
    LAYOUT_TAG="1n${CORES}c"
    LAYOUT_DESC="1 node, 1 task, ${CORES} cores, no mpirun"
    SBATCH_LAYOUT="#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CORES}
#SBATCH --exclusive"
else
    LAYOUT_TAG="${NODES}n${RANKS_PER_NODE}r"
    LAYOUT_DESC="${NODES} node(s) x ${RANKS_PER_NODE} ranks/node = ${RANKS_TOTAL} ranks"
    if [ "$MPI_SMC" = 1 ]; then
        # one core per rank and the whole node's memory: every rank compiles its
        # own model (about 0.8 GB here), so the rank count is what sets the
        # memory the job needs, and --mem=0 keeps a partition default from
        # capping it below what the node has
        SBATCH_LAYOUT="#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${RANKS_PER_NODE}
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive"
    else
        # MultiNest: the node is reserved and ppr places the ranks on it
        SBATCH_LAYOUT="#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}"
    fi
fi

[ -f parfile.par ] || { echo "parfile.par not found: run this from the experiment directory" >&2; exit 2; }
[ -d ../Inputs ] || { echo "../Inputs not found: run this from the experiment directory" >&2; exit 2; }

# the solver and the layout are part of the tag, so two runs never share a
# parfile, output or memory label
TAG="${RES_K}k_${LAYOUT_TAG}_${BRANCH}_${SOLVER_TAG}"
PARFILE="parfile_${TAG}.par"
OUTPUT="output_${TAG}.hdf5"
MULTI="multi_${TAG}"
LABEL="mem_${LAYOUT_TAG}_${RES_K}k_${BRANCH}_${SOLVER_TAG}"
JOB_NAME="taurex-${RES_K}k-${LAYOUT_TAG}-${BRANCH}-${SOLVER_TAG}"
LIGHT=${LIGHT:-1}
# the compiled programs are shared through JAX's persistent cache: without it
# every rank of a sharded run compiles the same program at the same time
JAX_CACHE_DIR=${JAX_CACHE_DIR:-${PWD}/.jax_cache}
mkdir -p "$JAX_CACHE_DIR" || { echo "cannot create $JAX_CACHE_DIR" >&2; exit 2; }
PARTITION=${PARTITION:-rome}

# ---------------- write the Slurm script that will be submitted ----------------
SCRIPT_DIR=${SCRIPT_DIR:-/tmp/taurex_submit}
mkdir -p "$SCRIPT_DIR"
SCRIPT="$SCRIPT_DIR/$(date +%Y-%m-%d_%H-%M-%S)_${JOB_NAME}.sh"

# ---------------- the command the job runs ----------------
LIGHT_FLAG=
[ "$LIGHT" = 1 ] && LIGHT_FLAG=--light

# the monitor reads the Pss of every process, so its thread count and interval
# stay small once there are many ranks to walk
MEM_THREADS=8 MEM_INTERVAL=0.05
[ "$MPI_SMC" = 1 ] && { MEM_THREADS=1; MEM_INTERVAL=0.5; }

if [ "$SOLVER" = jax ] && [ "$MPI_SMC" = 0 ]; then
    # --cpu-bind=none keeps Slurm from pinning the task to one core, which would
    # leave XLA a pool of one
    LAUNCH="    srun --exclusive -N1 -n1 --cpus-per-task=${CORES} --cpu-bind=none \\
        taurex -i ${PARFILE} --retrieval -o ${OUTPUT} ${LIGHT_FLAG}"
else
    LAUNCH="    mpirun --map-by ppr:${RANKS_PER_NODE}:node -np ${RANKS_TOTAL} \\
        taurex -i ${PARFILE} --retrieval -o ${OUTPUT} ${LIGHT_FLAG}"
fi

cat > "$SCRIPT" <<EOF
#!/bin/bash
# written by submit.sh on $(date -Is)
# ${LAYOUT_DESC}, R${RES_K}k, ${BRANCH} build
#SBATCH --job-name=${JOB_NAME}
${SBATCH_LAYOUT}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
$( [ -n "$EXCLUSIVE" ] && [ "$SOLVER" != jax ] && echo "#SBATCH --exclusive")

module purge
module load 2025
module load foss/2025b
module load Python/3.13.5-GCCcore-14.3.0

source ${VENV}/bin/activate
# TauRex looks for MultiNest under this path
export LD_LIBRARY_PATH=../MultiNest/lib:\$LD_LIBRARY_PATH
$( [ -n "$JAX_CACHE_DIR" ] && echo "export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}")
$( [ -n "$XLA_OPT_LEVEL" ] && echo "export XLA_FLAGS=\"--xla_backend_optimization_level=${XLA_OPT_LEVEL}\"")

# this run's parfile: its own cross-sections and chain path, and the solver's
# [Optimizer] section in place of the one in parfile.par
sed -e "s|^xsec_path = .*|xsec_path = ${XSEC_PATH}|" \\
    -e "s|^multi_nest_path = .*|multi_nest_path = ${MULTI}|" \\
    parfile.par | awk -v block="${JAX_BLOCK}" '
        block == "" { print; next }
        /^\[Optimizer\]/ {
            print
            n = split(block, key, "\n")
            for (i = 1; i <= n; i++) print key[i]
            skip = 1
            next
        }
        skip && /^\[/ { skip = 0 }
        skip { next }
        { print }
    ' > ${PARFILE}

# -t: threads that read Pss, -i: seconds between samples
taurex-mem-run -l ${LABEL} -i ${MEM_INTERVAL} -t ${MEM_THREADS} -- \\
${LAUNCH}
EOF
chmod +x "$SCRIPT"
ln -sf "$(basename "$SCRIPT")" "$SCRIPT_DIR/${JOB_NAME}.sh"

echo "solver   : $SOLVER"
if [ "$SOLVER" = jax ]; then
    echo "sampler  : $JAX_SAMPLER, samples=$JAX_NUM_SAMPLES"
    [ "$JAX_SAMPLER" = smc ] && echo "smc      : particles=$JAX_NUM_PARTICLES, hmc=$JAX_NUM_MCMC_STEPS x $JAX_SMC_HMC_STEPS, ess=$JAX_SMC_TARGET_ESS, delta=$JAX_SMC_MIN_DELTA..$JAX_SMC_MAX_DELTA"
fi
[ "$MPI_SMC" = 1 ] && echo "mpi      : ${RANKS_TOTAL} ranks, $JAX_NUM_PARTICLES particles"
echo "layout   : $LAYOUT_DESC"
echo "run tag  : $TAG"
echo "script   : $SCRIPT"

SUBMIT_OUT=$(sbatch "$SCRIPT")
echo "$SUBMIT_OUT"
sed -i "2i # $SUBMIT_OUT" "$SCRIPT"
