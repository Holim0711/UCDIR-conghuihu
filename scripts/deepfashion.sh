#!/bin/bash

method="method1"

python main.py \
    -a resnet50 \
    --batch-size 64 \
    --mlp --aug-plus --cos \
    --data-A '/datasets/DeepFashion1/Consumer_to_Shop_Retrieval' \
    --data-B '/datasets/DeepFashion1/Consumer_to_Shop_Retrieval' \
    --num-cluster '1024' \
    --warmup-epoch 5 \
    --temperature 0.2 \
    --exp-dir './experiments/method1/deepfashion-crop-withoutfc' \
    --lr 0.0002 \
    --clean-model 'moco_v2_800ep_pretrain.pth.tar' \
    --instcon-weight 1.0 \
    --cwcon-startepoch 5 \
    --cwcon-satureepoch 20 \
    --cwcon-weightstart 0.0 \
    --cwcon-weightsature 1.0 \
    --cwcon-filterthresh 0.2 \
    --epochs 100 \
    --selfentro-temp 0.1 \
    --selfentro-weight 0.5 \
    --selfentro-startepoch 20 \
    --distofdist-weight 0.1 \
    --distofdist-startepoch 20 \
    --prec-nums '1,20,50' \
    --withoutfc 'True' \
    --method $method \
    --gpu 2
