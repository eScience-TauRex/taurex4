#!/bin/bash
#SBATCH --job-name=taurex_retrieval
#SBATCH --output=taurex_%j.out
#SBATCH --error=taurex_%j.err
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Adjust based on your parallel processing needs

# Exit immediately if any command fails
set -e

echo "================ Job Started: $(date) ================"

echo "Loading modules..."
module purge  # Clear any conflicting inherited modules
module load 2025
module load GCC/14.3.0
module load OpenBLAS/0.3.30-GCC-14.3.0
module load OpenMPI/5.0.7-GCC-14.2.0


# Activate your local virtual environment
source .venv/bin/activate

# Run your local TauRex environment setup script
./setup_local_taurex_venv.sh

# Export the library path so TauRex can find MultiNest
export LD_LIBRARY_PATH=../MultiNest/lib:$LD_LIBRARY_PATH

# Run the simulation
taurex -i parfile_simple.par -o simple_retrieval.hdf5 --retrieval

echo "================ Job Finished: $(date) ================"
