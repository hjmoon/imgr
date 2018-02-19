#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV1 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v1_on_flowers.sh
set -e
#export CUDA_DEVICE_ORDER="PCI_BUS_ID"
#export CUDA_VISIBLE_DEVICES='1'

# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=../src/pretrained_ckpt

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=./tfmodels/inception_v3_train_v2


python train_tripletloss_gpu.py \
  --logs_base_dir=${TRAIN_DIR} \
  --models_base_dir=${TRAIN_DIR} \
  --model_def=inception_v3 \
  --max_nrof_epochs=100 \
  --batch_size=4 \
  --image_size=500 \
  --checkpoint_exclude_scopes=InceptionV3/AuxLogits,InceptionV3/Logits,Triplet \
  --alpha=0.4 \
  --embedding_size=1024 \
  --optimizer=SGD \
  --learning_rate=0.001 \
  --learning_rate_decay_factor=1.0 \
  --learning_rate_decay_epochs=10 \
  --log_interval=600 \
  --pretrained_model=${PRETRAINED_CHECKPOINT_DIR}/model.ckpt-337237 \
  --num_clone=1 \
  --num_bbox=34

