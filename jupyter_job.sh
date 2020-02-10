#!/bin/bash
#SBATCH -p cpu
#SBATCH --gres=gpu:0
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --job-name=test_jupyter
#SBATCH --output=jupyter_notebook_%j.log
#SBATCH --ntasks=1
#SBATCH --time=10000

date;hostname;pwd

cd $SLURM_SUBMIT_DIR
. jupyter.env
export XDG_RUNTIME_DIR=""
/h/arashwan/.conda/envs/fpncnets/bin/jupyter notebook --ip 0.0.0.0 --port 8888
