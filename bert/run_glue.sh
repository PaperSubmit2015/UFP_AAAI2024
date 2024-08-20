#!/bin/bash
#SBATCH -p q_intel_gpu_nvidia
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -o   bert-base.out

#SBATCH --ntasks=4
#SBATCH --job-name=bert-base

export  CUDA_HOME=/usr/local/cuda-11.8

module load  amd/cuda/11.8.89    amd/gcc_compiler/11.3.0
#python -u  dali.py -a  resnet18   -b  1024  -p 50  \
python -u  glue.py  \
          --model_name_or_path bert-base-uncased \
          --task_name cola  \
          --do_train --do_eval \
          --max_seq_length 128 \
          --per_device_train_batch_size 32 \
          --learning_rate 2e-5 \
          --num_train_epochs 3 \
          --overwrite_output_dir \
          --output_dir ./res/cola/    


      
       #--qam   eciz    --qad 8  \
       #--qwm   eciz    --qwd 8  \
       # /home/export/base/ycsc_rdc/thudev/online1 
      # --pretrained   --lr 0.00005  --epochs=50 \
      #  --train_with_eci 1  \
      # --qam    e3m4  --qad 8  \
      # --qwm    e3m4  --qwd 8  \
