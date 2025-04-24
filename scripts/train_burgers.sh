#!/bin/bash
#SBATCH --gres=gpu:1        # total number of GPUs
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G           # host memory per CPU core (node)
#SBATCH --time=0-04:00      # time (DD-HH:MM)
#SBATCH --account=def-soulaima
#SBATCH --job-name=b1d
#SBATCH --output=outfiles/b1d.out

# Load proper GPU env variables
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# Modify memory allocation behavior
# export XLA_PYTHON_CLIENT_PREALLOCATE=false     # Disable preallocation
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45     # Uncomment to set preallocation to x%
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform   # Uncomment to allocate/deallocate on demand (Use with curriculum training))

# Change to dir where file is located
cd /project/def-soulaima/mmullins/SWE-PINN/burgers

# Run program
python ./main.py

# To run  job script:
# In SWE-PINN, run the following command: sbatch scripts/train_burgers.sh