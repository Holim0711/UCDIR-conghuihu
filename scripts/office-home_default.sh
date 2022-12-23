#!/bin/bash

domainsA=("Art" "Clipart" "Product" "Real")
domainsB=("Art" "Clipart" "Product" "Real")

method="default"

for domainA in ${domainsA[@]};
do
    for domainB in ${domainsB[@]};
    do
        if [ $domainA != $domainB ]
        then
            tgdomainA=$domainA          
            tgdomainB=$domainB
            if [ $domainA == "Real" ]
            then
                tgdomainA="Real World"
            fi
            if [ $domainB == "Real" ]
            then 
                tgdomainB="Real World"
            fi
            python main.py \
                -a resnet50 \
                --batch-size 64 \
                --mlp --aug-plus --cos \
                --data-A "/datasets/OfficeHome/images/$tgdomainA" \
                --data-B "/datasets/OfficeHome/images/$tgdomainB" \
                --num-cluster '65' \
                --warmup-epoch 20 \
                --temperature 0.2 \
                --exp-dir "./experiments/$method/officehome/$tgdomainA-$tgdomainB" \
                --lr 0.0002 \
                --clean-model 'moco_v2_800ep_pretrain.pth.tar' \
                --instcon-weight 1.0 \
                --cwcon-startepoch 20 \
                --cwcon-satureepoch 100 \
                --cwcon-weightstart 0.0 \
                --cwcon-weightsature 0.5 \
                --cwcon-filterthresh 0.2 \
                --epochs 200 \
                --selfentro-temp 0.01 \
                --selfentro-weight 1.0 \
                --selfentro-startepoch 100 \
                --distofdist-weight 0.5 \
                --distofdist-startepoch 100 \
                --prec-nums '1,5,15,20' \
                --withoutfc 'True' \
                --method $method \
                --gpu 3
        fi
    done
done
