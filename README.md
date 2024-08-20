# UFP_AAAI2025
Here is the source code for AAAI2025.




# ResNet 18/50

The evaluations are tested on: Python=3.10,  Torch=2.1.0+cu121,  Torchvision=0.16.0+cu121 ,  Numpy=1.25, gcc=10.2.
Ninja JIT compiler is needed.

Some software version may cause conflicts, such as Torch>=2.4 and Numpy=1.x. 

## 1. Setup Environment

      conda create --name resnet  python==3.10
      conda activate resnet
      
      pip install torch==2.1.0 
      pip install numpy==1.25.2
      pip install torchvision=0.16
   
      pip install cupy-cuda11x
      pip install cupy-cuda12x
      pip install matplotlib
      pip install lmdb
      pip install nvidia-dali-cuda120
      pip install scipy
      pip install nvidia-ml-py
      pip install nvidia-nvjpeg2k-cu12

## 2. Train Model

      python -u  dali.py  -a  resnet18   -b 640  -p 400   --epochs 90   --qcfg  r18ufp.json  imagenet

 We provide 3 configuration scripts (r18fp32.json, r18eci.json, f18ufp.json) for FP32, FP8, and UFP training. You can replace the --qconf parameter to enable different training mode.
 

# ViT  

Pytorch reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.

This paper show that Transformers applied directly to image patches and pre-trained on large datasets work really well on image recognition task.

Vision Transformer achieve State-of-the-Art in image recognition task with standard Transformer encoder and fixed-size patches. In order to perform classification, author use the standard approach of adding an extra learnable "classification token" to the sequence.

The evaluations are tested on: Python=3.10,  Torch=2.1.0+cu121,  Torchvision=0.16.0+cu121 ,  Numpy=1.25, gcc=10.2.
Ninja JIT compiler is needed.

Some software version may cause conflicts, such as Torch>=2.4 and Numpy=1.x. 

## 1. Setup Environment

    conda create -n vit  python==3.10

    pip install   torch==2.1
    pip install   torchvision==0.16
    pip install   numpy==1.25
    pip install   tqdm
    pip install   tensorboard
    pip install   ml-collections
    pip install   packaging
   
    git clone https://www.github.com/nvidia/apex
    cd apex
    python setup.py install   


## 2. Download Pre-trained model (Google's Official Checkpoint)
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**), R50+ViT-B_16(**97.96M**), ViT-B_32(**87.5M**), ViT-L_16(**303.4M**), ViT-L_32(**305.5M**), ViT-H_14(**630.8M**)
  * imagenet21k pre-train models
    * ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14
  * imagenet21k pre-train + imagenet2012 fine-tuned models
    * ViT-B_16-224, ViT-B_16, ViT-B_32, ViT-L_16-224, ViT-L_16, ViT-L_32
  * Hybrid Model([Resnet50](https://github.com/google-research/big_transfer) + Transformer)
    * R50-ViT-B_16

     
```
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz

# imagenet21k pre-train + imagenet2012 fine-tuning
wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz

```

## 3. Train Model
```
python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
```
CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).


## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [Pytorch Image Models(timm)](https://github.com/rwightman/pytorch-image-models)






# BERT 

The evaluations are tested on: Python=3.10,  Torch=2.1.0+cu121,   Numpy=1.25, gcc=10.2.
Ninja JIT compiler is needed.

Some software version may cause conflicts, such as Torch>=2.4 and Numpy=1.x. 


## 1. Setup 

[Hugging Face](https://huggingface.co/)

## 2. Usage 

### CoLA 
    export TASK_NAME=cola
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1 
    module load   compiler/gcc/10.2.0 
    python -u  glue.py  \
           --model_name_or_path bert-large-uncased \ 
           --task_name  $TASK_NAME  --with_tracking --max_length 128 \ 
           --per_device_train_batch_size 32 --learning_rate 2e-5 \
           --num_train_epochs 10 --output_dir ./res/$TASK_NAME/ 
           
### MRPC 
    export TASK_NAME=mrpc
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1 
    module load   compiler/gcc/10.2.0 
    python -u  glue.py  \
           --model_name_or_path bert-large-uncased \ 
           --task_name  $TASK_NAME  --with_tracking --max_length 128 \ 
           --per_device_train_batch_size 32 --learning_rate 2e-5 \
           --num_train_epochs 4 --output_dir ./res/$TASK_NAME/ 

### QNLI
    export TASK_NAME=qnli
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1 
    module load   compiler/gcc/10.2.0 
    python -u  glue.py  \
           --model_name_or_path bert-large-uncased \ 
           --task_name  $TASK_NAME  --with_tracking --max_length 128 \ 
           --per_device_train_batch_size 32 --learning_rate 2e-5 \
           --num_train_epochs 10 --output_dir ./res/$TASK_NAME/ 

### QQP
    export TASK_NAME=qqp
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1 
    module load   compiler/gcc/10.2.0 
    python -u  glue.py  \
           --model_name_or_path bert-large-uncased \ 
           --task_name  $TASK_NAME  --with_tracking --max_length 128 \ 
           --per_device_train_batch_size 32 --learning_rate 2e-5 \
           --num_train_epochs 10 --output_dir ./res/$TASK_NAME/ 

### RTE 
    export TASK_NAME=RTE
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1 
    module load   compiler/gcc/10.2.0 
    python -u  glue.py  \
           --model_name_or_path bert-large-uncased \ 
           --task_name  $TASK_NAME  --with_tracking --max_length 128 \ 
           --per_device_train_batch_size 32 --learning_rate 2e-5 \
           --num_train_epochs 10 --output_dir ./res/$TASK_NAME/ 
           
### STSB 
    export TASK_NAME=stsb
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1 
    module load   compiler/gcc/10.2.0 
    python -u  glue.py  \
           --model_name_or_path bert-large-uncased \ 
           --task_name  $TASK_NAME  --with_tracking --max_length 128 \ 
           --per_device_train_batch_size 32 --learning_rate 2e-5 \
           --num_train_epochs 10 --output_dir ./res/$TASK_NAME/ 

### SST2 
    export TASK_NAME=sst2
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1 
    module load   compiler/gcc/10.2.0 
    python -u  glue.py  \
           --model_name_or_path bert-large-uncased \ 
           --task_name  $TASK_NAME  --with_tracking --max_length 128 \ 
           --per_device_train_batch_size 32 --learning_rate 2e-5 \
           --num_train_epochs 10 --output_dir ./res/$TASK_NAME/ 

           
# GPT2

 The source code is taken from  the paper LoRA: Low-Rank Adaptation of Large Language Models.
 The official source code is runing on torch=1.7.1 and transformers=3.3.1.
 Our evaluations are tested on: Python=3.9.17,  Torch=2.1.0+cu121,   Numpy=1.25, gcc=10.2.
 Ninja JIT compiler is needed.

## 1. Setup 
      conda create --name gpt2  python==3.9.17
      conda activate gpt2
      
      pip install torch==2.1
      pip install transformers==4.35.2
      pip install spacy
      pip install progress
      pip install scipy
      

## 2. Installing `loralib` 
 ```
 cd loralib 
 pip install -e .
 ```
## 3. Train Model 


There are some common parameter setting for GPT2 training. 

```
## GPT MODEL

GPTMODEL could be set as large or medium to use  GPTCARD  gpt2.lg or gpt2.md

##  PEFT

FINETUNE mode could be set as  FULL or LORA.
For full parameter fine-tuning , set FINETUNE as FULL; 
For LoRA PEFT , set FINETUNE as LORA; 


## Precision 

For FP32 trainging,  set  --qcfg    as  gpt2fp32.json.
For FP8 trainging,  set  --qcfg    as  gpt2fp8.json.
For UFP8 trainging,  set  --qcfg    as  gpt2ufp.json.
```





###  DART

The training recipe is shown as below. 

    
    module load   compiler/gcc/10.2.0 
    
    start=$(date)
    
    export TRAIN_STEP=1
    export BEAM_STEP=1
    export DECODE_STEP=1
    export EVAL_STEP=1
    
    export FINETUNE=LORA
    export PRECISION=FP32
    export GPTMODEL=large
    export DATASET=dart
    export CUDADEV=0
    export LR=0.0002
    export EPOCH=10
    export BATCH=6
    let    PORT=29356+10*$CUDADEV
    export TRAINED_MODEL_NAME=model.104440.pt
    export DIR=./trained_models/GPT2_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION
    export PREDICT_FILENAME=$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION\_predict.jsonl
    let    PORT_BEAM=$PORT+2
    
    export  RANDOMSEED=110
    export ERROR=0
    
    if [ "$FINETUNE" = "LORA" ]; then 
        export LORA_DIM=4
        export LR=0.0002
        export EPOCH=5
        export BATCH=8
        export TRAINED_MODEL_NAME=model.39165.pt
    else
        export LORA_DIM=0
        export LR=0.00001
        export EPOCH=10
        export BATCH=6
        export TRAINED_MODEL_NAME=model.104440.pt
    fi
    
    if [ "$PRECISION" = "ECI" ]; then
        export QM=eciz
    elif [ "$PRECISION" = "FP32" ]; then
        export QM=none
    else
        let ERROR=$ERROR+1
        echo "[X] PRECESION is not supported!! "
    fi
    
    if    [ "$GPTMODEL" = "large" ]; then
        export GPTCARD=gpt2.lg
    elif  [ "$GPTMODEL" = "medium" ]; then
        export GPTCARD=gpt2.md
    else
        let ERROR=$ERROR+1
        echo "[X] PRECESION is not supported!! "
    fi
    
    
    if  [ "$ERROR" != 0 ];  then
        echo $ERROR  "errors occurs!. Now exits."
        exit
    fi
    



    if [ "$TRAIN_STEP" = 1 ]; then 
    
    rm $DIR/* 
    
    CUDA_VISIBLE_DEVICES=$CUDADEV \
      python -m torch.distributed.launch --master_port=$PORT --nproc_per_node=1 src/gpt2_ft.py  \
    --train_data ./data/$DATASET/train.jsonl \
    --valid_data ./data/$DATASET/valid.jsonl \
    --train_batch_size $BATCH \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card $GPTCARD \
    --init_checkpoint ./pretrained_checkpoints/gpt2-$GPTMODEL-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr $LR \
    --weight_decay 0.00 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch $EPOCH \
    --save_interval 8000 \
    --lora_dim  $LORA_DIM \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --label_smooth 0.0 \
    --random_seed $RANDOMSEED \
    --work_dir  $DIR  \
    --qcfg  gpt2fp8.json
    
    fi
    
    if [ "$BEAM_STEP" = 1 ]; then 
    CUDA_VISIBLE_DEVICES=$CUDADEV  \
    python -m torch.distributed.launch --nproc_per_node=1   --master_port=$PORT_BEAM  src/gpt2_beam.py  \
        --data ./data/$DATASET/test.jsonl \
        --batch_size 1 \
        --seq_len 512 \
        --eval_len 64 \
        --model_card $GPTCARD \
        --init_checkpoint $DIR/$TRAINED_MODEL_NAME \
        --platform local \
        --lora_dim 4 \
        --lora_alpha 32 \
        --beam 10 \
        --length_penalty 0.8 \
        --no_repeat_ngram_size 4 \
        --repetition_penalty 1.0 \
        --eos_token_id 628 \
        --work_dir $DIR \
        --output_file  $PREDICT_FILENAME
    fi
    
    
    if [ "$DECODE_STEP" = 1 ]; then 
    
    CUDA_VISIBLE_DEVICES=$CUDADEV  \
    python src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file $DIR/$PREDICT_FILENAME \
        --input_file ./data/$DATASET/test_formatted.jsonl \
        --ref_type dart \
        --ref_num 6 \
        --output_ref_file  eval/GenerationEval/data/references_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION  \
        --output_pred_file eval/GenerationEval/data/hypothesis_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION  \
        --tokenize --lower
    fi
    
    if [ "$EVAL_STEP" = 1 ]; then 
    cd ./eval/GenerationEval/
    python eval.py \
        -R data/references_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION/reference \
        -H data/hypothesis_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION   \
        -nr 6 \
        -m bleu,meteor,ter 
    cd ../..
    fi
    
    finish=$(date)
    
    echo  "Started : "  $start
    echo  "Finished: "  $finish   
    echo  $GPTMODEL  $DATASET  $FINETUNE  $PRECISION  $QM   $RANDOMSEED









###  WebNLG

The training recipe is shown as below. 


    
    module load   compiler/gcc/10.2.0 
    
    start=$(date)
    
    export NEED_TRAIN=1
    
    export FINETUNE=FULL
    export PRECISION=ECI
    export GPTMODEL=medium
    export DATASET=webnlg_challenge_2017
    export CUDADEV=0
    export LR=0.0002
    export EPOCH=10
    export BATCH=6
    export PORT=29596
    export TRAINED_MODEL_NAME=model.30050.pt
    export DIR=./trained_models/GPT2_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION
    export PREDICT_FILENAME=$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION\_predict.jsonl
    let    PORT_BEAM=$PORT+2
    
    export  RANDSEED=24680
    
    if [ "$FINETUNE" = "LORA" ]; then 
        export LORA_DIM=4
        export BATCH=8
        export EPOCH=5
        export LR=0.0002
        export TRAINED_MODEL_NAME=model.11270.pt
    else
        export LORA_DIM=0
        export BATCH=6
        export EPOCH=10
        export LR=0.00001
        export TRAINED_MODEL_NAME=model.30050.pt
    fi
    
    if [ "$PRECISION" = "ECI" ]; then
        export QM=eciz
    else 
        export QM=none
    fi
    
    if    [ "$GPTMODEL" = "large" ]; then
        export GPTCARD=gpt2.lg
    elif  [ "$GPTMODEL" = "medium" ]; then
        export GPTCARD=gpt2.md
    else
        export GPTCARD=gpt2.md
    fi
    
    if   [ "$CUDADEV" = 0 ]; then 
        export PORT=29556
    elif [ "$CUDADEV" = 1 ]; then 
        export PORT=29656
    else 
        export PORT=29756
    fi
    
    
    
    # train 
    if [ "$NEED_TRAIN" = 1 ] ; then
    
    rm $DIR  -r -f
    
    CUDA_VISIBLE_DEVICES=$CUDADEV \
      python -m torch.distributed.launch --master_port=$PORT --nproc_per_node=1 src/gpt2_ft.py  \
    --train_data ./data/$DATASET/train.jsonl \
    --valid_data ./data/$DATASET/valid.jsonl \
    --train_batch_size $BATCH \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card $GPTCARD \
    --init_checkpoint ./pretrained_checkpoints/gpt2-$GPTMODEL-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr $LR \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch $EPOCH \
    --save_interval 5000 \
    --lora_dim  $LORA_DIM \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --random_seed $RANDSEED  \
    --work_dir  $DIR  \
    --qcfg  gpt2fp8.json 
    
    
    fi
    
    CUDA_VISIBLE_DEVICES=$CUDADEV  \
    python -m torch.distributed.launch --nproc_per_node=1   --master_port=$PORT_BEAM  src/gpt2_beam.py  \
        --data ./data/$DATASET/test.jsonl \
        --batch_size 1 \
        --seq_len 512 \
        --eval_len 64 \
        --model_card $GPTCARD \
        --init_checkpoint $DIR/$TRAINED_MODEL_NAME \
        --platform local \
        --lora_dim 4 \
        --lora_alpha 32 \
        --beam 10 \
        --length_penalty 0.8 \
        --no_repeat_ngram_size 4 \
        --repetition_penalty 1.0 \
        --eos_token_id 628 \
        --work_dir $DIR \
        --output_file  $PREDICT_FILENAME
    
    
    CUDA_VISIBLE_DEVICES=$CUDADEV  \
    python src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file $DIR/$PREDICT_FILENAME \
        --input_file ./data/$DATASET/test_formatted.jsonl \
        --ref_type webnlg \
        --ref_num 6 \
        --output_ref_file  eval/GenerationEval/data/references_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION  \
        --output_pred_file eval/GenerationEval/data/hypothesis_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION  \
        --tokenize --lower
    
    cd ./eval/GenerationEval/
    python eval.py \
        -R data/references_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION/reference \
        -H data/hypothesis_$GPTMODEL\_$DATASET\_$FINETUNE\_$PRECISION   \
        -nr 6 \
        -m bleu,meteor,ter 
    cd ../..
    
    finish=$(date)
    
    echo  "Started : "  $start
    echo  "Finished: "  $finish   
    echo  "GPT model=" $GPTMODEL  "Dataset=" $DATASET  "FT=" $FINETUNE  "Precision=" $PRECISION  "Q method=" $QM
    echo  "lR=" $LR  "batch=" $BATCH   "epoch=" $EPOCH  
    
    
    
    
    
    

###  E2E

The training recipe is shown as below. 

    export QM=fullft_eci
    export GPTMODEL=large
    export GPTCARD=gpt2.lg
    
    
    CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --master_port=29508 --nproc_per_node=1 src/gpt2_ft.py  \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card $GPTCARD \
    --init_checkpoint ./pretrained_checkpoints/gpt2-$GPTMODEL-pytorch_model.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 3000 \
    --lora_dim  0 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --random_seed 110 \
    --work_dir ./trained_models/GPT2_$GPTMODEL\_$QM/e2e \
    --random_seed 110  \
    --qconf  gpt2ufp.json


    export DIR=./trained_models/GPT2_$GPTMODEL\_$QM/e2e 

    CUDA_VISIBLE_DEVICES=0  \
    python -m torch.distributed.launch --nproc_per_node=1   --master_port=29500  src/gpt2_beam.py  \
        --data ./data/e2e/test.jsonl \
        --batch_size 1 \
        --seq_len 512 \
        --eval_len 64 \
        --model_card $GPTCARD \
        --init_checkpoint ./trained_models/GPT2_$GPTMODEL\_$QM/e2e/model.26290.pt \
        --platform local \
        --lora_dim 4 \
        --lora_alpha 32 \
        --beam 10 \
        --length_penalty 0.8 \
        --no_repeat_ngram_size 4 \
        --repetition_penalty 1.0 \
        --eos_token_id 628 \
        --work_dir ./trained_models/GPT2_$GPTMODEL\_$QM/e2e \
        --output_file predict.26290.b10p08r4.jsonl


    CUDA_VISIBLE_DEVICES=1  \
    python src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file $DIR/predict.26290.b10p08r4.jsonl \
        --input_file ./data/e2e/test_formatted.jsonl \
        --output_ref_file $DIR/e2e_ref_$QM.txt \
        --output_pred_file $DIR/e2e_pred_$QM.txt

    CUDA_VISIBLE_DEVICES=1 \
    python eval/e2e/measure_scores.py $DIR/e2e_ref_$QM.txt $DIR/e2e_pred_$QM.txt -p






