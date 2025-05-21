#!/bin/bash
#SBATCH --job-name=pointnet_train
#SBATCH --partition=earth-4
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2


module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load slurm/slurm/21.08.6

source /cfs/earth/scratch/nogernic/venv/pointnet_env/bin/activate

cd /cfs/earth/scratch/nogernic/Pointnet2_PyTorch

# Make sure Unix line endings:
dos2unix test_pointnetpp.py

python test_pointnetpp.py
