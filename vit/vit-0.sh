
module load compiler/gcc/10.2.0
CUDA_VISIBLE_DEVICES=0    python -u  vit.py  \
      --name vitb32_ft  --img_size   384 \
      --train_batch_size  256   --eval_every 500  --num_steps 20000 \
      --output_dir ./logs  \
      --pretrained_dir  imagenet21k_ViT-B_32.npz   --model_type  ViT-B_32  \
      --dataset imagenet   \
      --qconf  vitufp.json \
      --statInterval  500  

  #    --qwm   eciz    --qwd 8  \
    


       # --fp16
       # --pretrained   --lr 0.00005  --epochs=50 \
       #  --train_with_eci 1  \

       #--qgm   eciz    --qgd 8  \
       #--qam   eciz    --qad 8  \
       #--qwm   eciz    --qwd 8  \
       #--resume  resnet18_59.pth.tar \

