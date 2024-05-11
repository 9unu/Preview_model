#!/bin/bash

fp="./resources_ocr/data/train/"
log_path="./logs_ocr/"
log_filename="pos_analysis.log"
encoding=utf-8-sig

python ./src_ocr/do_posTagging.py --fp=$fp --log_fp=$log_path --log_filename=$log_filename --encoding=$encoding