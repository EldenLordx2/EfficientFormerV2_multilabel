#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
MODEL=efficientformerv2_l
nGPUs=4

nohup python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --model $MODEL \
  --data-set TXT \
  --train-txt train.txt \
  --val-txt val.txt \
  --label-txt label.txt \
  --output_dir efficientformer_res1 \
  --save-freq 10 \
  --epochs 50 \
  --lr 2e-3\
  --warmup-lr 1e-3 \
  --min-lr 1e-4 \
  > output.log  2>&1 &
