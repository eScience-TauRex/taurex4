#!/bin/bash
#
# One submitter for the retrievals in this directory.
#
#   ./submit.sh <nodes> <ranks-per-node> <resolution-k> [branch] [solver]
#
#   nodes             nodes to allocate (1 for solver=jax)
#   ranks-per-node    MPI ranks per node, or the cores of the one jax task
#   resolution-k      cross-section resolution in k: 1, 15, 50, 100
#   branch            jax (taurex4, .venv_jax, default) | original (taurex3, .venv)
#   solver            multinest [default] | jax | laplace | nuts | mclmc | smc
#
# solver=jax is the differentiability plugin's optimizer: it fits the mode and
# draws with the sampler from the solver argument (laplace, nuts, mclmc, smc;
# smc also returns log Z). It always runs as one task on one node, with XLA
# spreading the compiled programs over <ranks-per-node> cores. multinest runs
# taurex3 under mpirun.
#
# The hyperparameters below are set for one final posterior on the 11-parameter
# WFC3 fit: a short mode fit for the samplers, 1000 NUTS draws, and 512 SMC
# particles started from the mode so the proposal does the work. Compiled
# programs are cached in $JAX_CACHE_DIR, so only the first run pays the minutes
# of compilation.
#
# Environment overrides:
#   JAX_NUM_SAMPLES=1000    nuts/mclmc draws
#   JAX_NUM_CHAINS=16       nuts/mclmc chains
#   JAX_NUM_WARMUP=200      nuts warm-up steps per chain
#   JAX_NUM_PARTICLES=512   smc particles; also its posterior resolution
#   JAX_SMC_HMC_STEPS=8     smc leapfrogs per particle per stage
#   JAX_SMC_BATCH_SIZE=32   smc particles per vmapped batch (memory knob)
#   JAX_CACHE_DIR=$PWD/.jax_cache   exported as JAX_COMPILATION_CACHE_DIR
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
    jax) JAX_SAMPLER=${JAX_SAMPLER:-laplace} ;;
    multinest) ;;
    *) echo "unknown solver: $SOLVER (known: multinest, jax, laplace, nuts, mclmc, smc)" >&2; exit 2 ;;
esac
if [ "$SOLVER" = jax ]; then
    case "$JAX_SAMPLER" in
        laplace|nuts|mclmc|smc) ;;
        *) echo "unknown sampler: $JAX_SAMPLER (known: laplace, nuts, mclmc, smc)" >&2; exit 2 ;;
    esac
    [ "$BRANCH" = jax ] || { echo "solver jax needs branch jax" >&2; exit 2; }
fi

# ---------------- sampler budget ----------------
JAX_NUM_SAMPLES=${JAX_NUM_SAMPLES:-1000}
JAX_NUM_CHAINS=${JAX_NUM_CHAINS:-16}
JAX_NUM_WARMUP=${JAX_NUM_WARMUP:-200}
JAX_NUM_PARTICLES=${JAX_NUM_PARTICLES:-512}
JAX_SMC_HMC_STEPS=${JAX_SMC_HMC_STEPS:-8}
JAX_SMC_BATCH_SIZE=${JAX_SMC_BATCH_SIZE:-32}

# The mode is the product only for laplace; for a sampler it just seeds the
# chain or the proposal, so it stops at a looser gradient and fewer steps.
case "$JAX_SAMPLER" in
    laplace) JAX_FIT_ITERATIONS=200; JAX_GRAD_TOL=1e-10 ;;
    *)       JAX_FIT_ITERATIONS=100; JAX_GRAD_TOL=1e-4 ;;
esac

SOLVER_TAG=$SOLVER
JAX_BLOCK=
if [ "$SOLVER" = jax ]; then
    SOLVER_TAG=jax_${JAX_SAMPLER}
    JAX_BLOCK="optimizer = jax
sampler = ${JAX_SAMPLER}
num_samples = ${JAX_NUM_SAMPLES}
max_iterations = ${JAX_FIT_ITERATIONS}
gradient_tolerance = ${JAX_GRAD_TOL}"
    case "$JAX_SAMPLER" in
        nuts)
            JAX_BLOCK="${JAX_BLOCK}
num_chains = ${JAX_NUM_CHAINS}
num_warmup = ${JAX_NUM_WARMUP}" ;;
        mclmc)
            JAX_BLOCK="${JAX_BLOCK}
num_chains = ${JAX_NUM_CHAINS}" ;;
        smc)
            # Start from the mode so the Student-t proposal carries the
            # posterior: when its weights pass the ESS and k-hat checks the run
            # spends no gradients at all, otherwise one HMC polish at beta=1
            # corrects it. The prior start is only the fallback.
            JAX_BLOCK="${JAX_BLOCK}
num_particles = ${JAX_NUM_PARTICLES}
num_mcmc_steps = 1
smc_hmc_steps = ${JAX_SMC_HMC_STEPS}
smc_start = laplace
smc_metric = auto
smc_fit = lbfgs
smc_max_iterations = ${JAX_FIT_ITERATIONS}
smc_target_ess = 0.5
smc_polish_ess = 0.15
smc_max_delta = 0.5
smc_max_steps = 200
smc_step_size = 0.5
smc_batch_size = ${JAX_SMC_BATCH_SIZE}" ;;
    esac
    JAX_BLOCK="${JAX_BLOCK}
"
fi

# the plugin and the sampler's library are needed after the mode fit, so a
# missing one would only show up at the end of a long job
if [ "$SOLVER" = jax ]; then
    [ -x "$VENV/bin/python" ] || { echo "$VENV/bin/python not found" >&2; exit 2; }
    MODS=taurex.differentiability
    [ "$JAX_SAMPLER" = smc ] && MODS="$MODS blackjax.smc.tempered" || MODS="$MODS blackjax"
    for m in $MODS; do
        "$VENV/bin/python" -c "import $m" >/dev/null \
            || { echo "$VENV cannot import $m: install the jax deps" >&2; exit 2; }
    done
fi

# ---------------- launch layout ----------------
# solver=jax is one task on one node, with <ranks-per-node> read as its core
# count: XLA spreads every compiled program over those cores.
NTASKS_PER_NODE=128
TIME=${TIME:-03:00:00}
[ "$RANKS_PER_NODE" -le "$NTASKS_PER_NODE" ] || {
    echo "$RANKS_PER_NODE ranks/node does not fit in a node of $NTASKS_PER_NODE cores" >&2
    exit 2
}

[ -f parfile.par ] || { echo "parfile.par not found: run this from the experiment directory" >&2; exit 2; }
[ -d ../Inputs ] || { echo "../Inputs not found: run this from the experiment directory" >&2; exit 2; }

# the solver and the layout are part of the tag, so two runs never share a
# parfile, output or memory label
JAX_CACHE_DIR=${JAX_CACHE_DIR:-${PWD}/.jax_cache}
mkdir -p "$JAX_CACHE_DIR" || { echo "cannot create $JAX_CACHE_DIR" >&2; exit 2; }
LIGHT=${LIGHT:-1}
PARTITION=${PARTITION:-rome}
SCRIPT_DIR=${SCRIPT_DIR:-/tmp/taurex_submit}
mkdir -p "$SCRIPT_DIR"
LIGHT_FLAG=
[ "$LIGHT" = 1 ] && LIGHT_FLAG=--light

if [ "$SOLVER" = jax ]; then
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
#SBATCH --mem=0
#SBATCH --exclusive"
else
    RANKS_TOTAL=$((NODES * RANKS_PER_NODE))
    LAYOUT_TAG="${NODES}n${RANKS_PER_NODE}r"
    LAYOUT_DESC="${NODES} node(s) x ${RANKS_PER_NODE} ranks/node = ${RANKS_TOTAL} ranks"
    SBATCH_LAYOUT="#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}"
fi

TAG="${RES_K}k_${LAYOUT_TAG}_${BRANCH}_${SOLVER_TAG}"
PARFILE="parfile_${TAG}.par"
OUTPUT="output_${TAG}.hdf5"
MULTI="multi_${TAG}"
LABEL="mem_${LAYOUT_TAG}_${RES_K}k_${BRANCH}_${SOLVER_TAG}"
JOB_NAME="taurex-${RES_K}k-${LAYOUT_TAG}-${BRANCH}-${SOLVER_TAG}"

if [ "$SOLVER" = jax ]; then
    # --cpu-bind=none keeps Slurm from pinning the task to one core, which would
    # leave XLA a pool of one
    LAUNCH="    srun --exclusive -N1 -n1 --cpus-per-task=${CORES} --cpu-bind=none \\
        taurex -i ${PARFILE} --retrieval -o ${OUTPUT} ${LIGHT_FLAG}"
else
    LAUNCH="    mpirun --map-by ppr:${RANKS_PER_NODE}:node -np ${RANKS_TOTAL} \\
        taurex -i ${PARFILE} --retrieval -o ${OUTPUT} ${LIGHT_FLAG}"
fi

SCRIPT="$SCRIPT_DIR/$(date +%Y-%m-%d_%H-%M-%S)_${JOB_NAME}.sh"
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
# TauREx looks for MultiNest under this path
export LD_LIBRARY_PATH=../MultiNest/lib:\$LD_LIBRARY_PATH
# the ranks share compiled programs through this cache
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR}

# this run's parfile: its own cross-sections, chain path and [Optimizer] section
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

taurex-mem-run -l ${LABEL} -i 0.1 -t 8 -- \\
${LAUNCH}
EOF
chmod +x "$SCRIPT"
ln -sf "$(basename "$SCRIPT")" "$SCRIPT_DIR/${JOB_NAME}.sh"

echo "solver   : $SOLVER"
if [ "$SOLVER" = jax ]; then
    echo "sampler  : $JAX_SAMPLER"
    case "$JAX_SAMPLER" in
        smc) echo "smc      : ${JAX_NUM_PARTICLES} particles, ${JAX_SMC_HMC_STEPS} leapfrogs, batch ${JAX_SMC_BATCH_SIZE}, laplace start" ;;
        laplace) echo "draws    : ${JAX_NUM_SAMPLES} from the Laplace Gaussian" ;;
        *) echo "draws    : ${JAX_NUM_SAMPLES} over ${JAX_NUM_CHAINS} chains" ;;
    esac
fi
echo "layout   : $LAYOUT_DESC"
echo "run tag  : $TAG"
echo "script   : $SCRIPT"

SUBMIT_OUT=$(sbatch "$SCRIPT")
echo "$SUBMIT_OUT"
sed -i "2i # $SUBMIT_OUT" "$SCRIPT"
