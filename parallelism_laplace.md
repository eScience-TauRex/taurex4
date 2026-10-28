# Parallelism with the Laplace optimizer

Scope: `optimizer = laplace` (JAX path), one CPU node or a CPU cluster. Memory
from the reference input of `differentiability_report_jax.md` (80 layers, 426
point grid, 2000 draws): 0.81 GB peak for one `taurex -R` process.

- The parallelism is inside one process: XLA runs the compiled forward,
  gradient, Hessian and L-BFGS programs over the cores that process can see.
  The 200 quasi-Newton steps are a sequential chain.
- MPI ranks are separate processes with separate JAX runtimes; no array is
  sharded between them. One rank fits, the others take the mode, the draws, the
  covariance and the profile bands by one broadcast. Per node at 64 ranks that
  is 1 compiled model instead of 64, 1 compilation instead of 64, and 0.81 GB
  instead of 52 GB (64 x 0.81 GB).
- A single fit cannot leave its node, so on a cluster more cores per node help
  and more nodes do not. The numpy model, its warm-up and `generate_solution`
  still run on every rank, as they do for the sampling optimizers.
- Binding: `-np 64` with Open MPI's default `--bind-to core` gives the fitting
  rank one core, and XLA sizes its pool from `sched_getaffinity`. One task with
  the whole node to itself is faster than 64 ranks sharing it.
- Of the work that could be split over ranks, only independent retrievals pay:
  the 2000 draws already cost +0.03 GB, the Hessian has `nparams` columns, and
  sharding the wavenumber axis adds a collective to each of 200 steps. A fit is
  ~15 s end to end, 6.5 s of it compilation; a sharded fit has not been measured.

## Command

One retrieval, the whole node, no `mpirun` (there is nothing to reduce across
tasks):

```
srun --exclusive -N1 -n1 --cpus-per-task=<cores> --cpu-bind=none \
    taurex -i parfile.par --retrieval -o output.hdf5 --light
```

`--cpu-bind=none` keeps SLURM from pinning the single task to one core, and
`--exclusive` keeps other jobs off the node. If the fit looks single-threaded,
check the affinity mask of the running process with `taskset -c -p <pid>`.

The only way to use more nodes is more retrievals: one task and one node each,
e.g. a job array with `-N1 -n1` per element and a different `-i`/`-o` per
`SLURM_ARRAY_TASK_ID`. In this branch a multi-rank job is correct, but the fit
runs on one rank bound to one core, so `-np 128` buys the allocation and no
speed.

