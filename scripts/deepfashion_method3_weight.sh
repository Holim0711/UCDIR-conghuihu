#!/bin/bash

method="method3"
domain_weight=0.1

python main_3.py \
    -a resnet50 \
    --batch-size 64 \
    --mlp --aug-plus --cos \
    --data-A '/datasets/DeepFashion1/Consumer_to_Shop_Retrieval' \
    --data-B '/datasets/DeepFashion1/Consumer_to_Shop_Retrieval' \
    --num-cluster '100' \
    --warmup-epoch 5 \
    --temperature 0.2 \
    --exp-dir "./test/experiments/$method/$domain_weight/deepfashion-crop-withoutfc/2" \
    --lr 0.0002 \
    --clean-model 'moco_v2_800ep_pretrain.pth.tar' \
    --instcon-weight 1.0 \
    --cwcon-startepoch 5 \
    --cwcon-satureepoch 20 \
    --cwcon-weightstart 0.0 \
    --cwcon-weightsature 1.0 \
    --cwcon-filterthresh 0.2 \
    --epochs 30 \
    --selfentro-temp 0.1 \
    --selfentro-weight 0.5 \
    --selfentro-startepoch 20 \
    --distofdist-weight 0.1 \
    --distofdist-startepoch 20 \
    --prec-nums '1,20,50' \
    --withoutfc 'True' \
    --method $method \
    --domain-weight $domain_weight \
    --smg 'm' \
    --gpu 0
