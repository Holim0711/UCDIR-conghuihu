#!/bin/bash

domainsA=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
domainsB=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
method="method1"

for domainA in ${domainsA[@]};
do
    for domainB in ${domainsB[@]};
    do
        if [ $domainA != $domainB ]
        then
            python main.py \
                -a resnet50 \
                --batch-size 64 \
                --mlp --aug-plus --cos \
                --data-A "/datasets/DomainNet/images/$domainA" \
                --data-B "/datasets/DomainNet/images/$domainB" \
                --num-cluster '7' \
                --warmup-epoch 20 \
                --temperature 0.2 \
                --exp-dir "./experiments/$method/domainnet/$domainA-$domainB" \
                --lr 0.0002 \
                --clean-model 'moco_v2_800ep_pretrain.pth.tar' \
                --instcon-weight 1.0 \
                --cwcon-startepoch 20 \
                --cwcon-satureepoch 100 \
                --cwcon-weightstart 0.0 \
                --cwcon-weightsature 1.0 \
                --cwcon-filterthresh 0.2 \
                --epochs 200 \
                --selfentro-temp 0.1 \
                --selfentro-weight 0.5 \
                --selfentro-startepoch 100 \
                --distofdist-weight 0.1 \
                --distofdist-startepoch 100 \
                --prec-nums '20,50,100,200' \
                --withoutfc 'True' \
                --method $method \
                --gpu 0
        fi
    done
done
