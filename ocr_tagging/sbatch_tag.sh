#! /usr/bin/bash

#SBATCH -J p-review
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ugrad
#SBATCH -t 24:0:0
#SBATCH -w aurora-g7
#SBATCH -o logs/slurm-%A.out

# batch size down, train_fp, valid_fp change, out_model_path change
cd /data/ndn825/Preview_model/ocr_tagging

/data/$USER/anaconda3/bin/conda init
source ~/.bashrc
conda activate preview_model

echo "PWD"
pwd
echo "WHICH PYTHON"
which python
echo "HOSTNAME"
hostname

python ./src_ocr/do_tagging.py

exit 0
