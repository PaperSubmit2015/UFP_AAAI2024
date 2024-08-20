#!/bin/bash
#SBATCH -p q_intel_gpu_nvidia
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -o   bert-base.out

#SBATCH --ntasks=4
#SBATCH --job-name=bert-base

export  CUDA_HOME=/usr/local/cuda-11.8


export TASK_NAME=mrpc

#cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli

module load  amd/cuda/11.8.89    amd/gcc_compiler/11.3.0
#python -u  dali.py -a  resnet18   -b  1024  -p 50  \
python -u  glue.py  \
          --model_name_or_path bert-base-uncased \
          --task_name  $TASK_NAME  \
          --max_length 128 \
          --per_device_train_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 3 \
          --output_dir ./res/cola/    


      
       #--qam   eciz    --qad 8  \
       #--qwm   eciz    --qwd 8  \
       # /home/export/base/ycsc_rdc/thudev/online1 
      # --pretrained   --lr 0.00005  --epochs=50 \
      #  --train_with_eci 1  \
      # --qam    e3m4  --qad 8  \
      # --qwm    e3m4  --qwd 8  \
