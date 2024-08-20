# UFP_AAAI2024
Source code for AAAI2024


# ResNet 18/50
    Torch version: 2.1.0+cu121 
    Torchvision  version: 0.16.0+cu121 

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

## 1. Setup 

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




# GPT2



