#!/bin/bash
#SBATCH -t 0-3:00
#SBATCH --account=def-vmago
#SBATCH --mem=100000
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=12

module load python/3.6
source ~/ENV/bin/activate

module load scipy-stack

pip install --no-index scikit-learn
pip install --no-index -r requirements.txt
pip install --no-index scipy==1.1.0

 
python ./scripts/train_yt_script.py