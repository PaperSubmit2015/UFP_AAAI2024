#!/bin/bash
#SBATCH -p q_intel_gpu_nvidia
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -o   cola-base.out

#SBATCH --ntasks=4
#SBATCH --job-name=cola

#export  CUDA_HOME=/usr/local/cuda-11.8


export TASK_NAME=cola
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
export http_proxy=http://127.0.0.1:7899
export https_proxy=http://127.0.0.1:7899
module load   compiler/gcc/10.2.0

#cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli

#module load  amd/cuda/11.8.89    amd/gcc_compiler/11.3.0
#python -u  dali.py -a  resnet18   -b  1024  -p 50  \
CUDA_VISIBLE_DEVICES=0 \
python -u  glue.py  \
          --model_name_or_path bert-large-uncased \
          --task_name  $TASK_NAME  \
          --with_tracking \
          --max_length 128 \
          --per_device_train_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 10 \
          --output_dir ./res/$TASK_NAME/    \


      
