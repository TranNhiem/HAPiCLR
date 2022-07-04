python3 ../../../main_umap.py \
    --dataset imagenet \
    --data_dir /data1/1K_New \
    --train_dir train \
    --val_dir val \
    --subset_class_num 100\
    --batch_size 2048 \
    --num_workers 10 \
    --pretrained_checkpoint_dir /data1/MPLCL_ckpt/simclr/1pmubc5u/simclr-100ep-imagenet-1pmubc5u-ep=99.ckpt
