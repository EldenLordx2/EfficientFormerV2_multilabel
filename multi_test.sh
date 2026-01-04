#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
MODEL=efficientformerv2_l
nGPUs=1
CHECKPOINT=./efficientformer_res1/checkpoint_epoch9.pth

python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py \
  --model $MODEL \
  --resume $CHECKPOINT \
  --eval \
  --data-set TXT \
  --val-txt test.txt \
  --label-txt label.txt \
  --output_dir efficientformer_res1/test \
  --predict-only \
  --thr 0.5 \
  --pred-out pred_vec.txt \
  --pred-label-out pred_labels.txt
