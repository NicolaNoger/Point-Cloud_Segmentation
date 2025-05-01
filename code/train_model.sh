#!/bin/bash
#SBATCH --job-name=pointnet_train
#SBATCH --partition=earth-4
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2

source /cfs/earth/scratch/klaveflo/venvs/PA1/bin/activate

cd /cfs/earth/scratch/klaveflo/PA1

python code/model_hpc.py


