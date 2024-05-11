#!/bin/bash

epochs=10
train_batch_size=4
valid_batch_size=4
init_model_path=klue/bert-base
max_length=512
need_birnn=false
aspect_drop_ratio=0.3
aspect_in_feature=768
stop_patience=3
train_fp="./resources_ocr/data/train/"
valid_fp="./resources_ocr/data/valid/"
base_path="./ckpt_ocr/model/"
label_info_file="meta.bin"
out_model_path="pytorch_model.bin"


python ./src_ocr/do_train.py --epochs=$epochs --init_model_path=$init_model_path --train_batch_size=$train_batch_size --valid_batch_size=$valid_batch_size --max_length=$max_length --need_birnn=$need_birnn  --aspect_drop_ratio=$aspect_drop_ratio  --aspect_in_feature=$aspect_in_feature --stop_patience=$stop_patience --train_fp=$train_fp --valid_fp=$valid_fp --base_path=$base_path --label_info_file=$label_info_file --out_model_path=$out_model_path

