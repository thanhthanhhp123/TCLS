#!/bin/sh
#SBATCH --job-name=code-thanh
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thanhmaxdz2003@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load cuda
module load cudnn
eval "$(conda shell.bash hook)"
conda activate thuy
python3 main.py
