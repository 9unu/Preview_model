#!/bin/bash

fp="./resources_review/data/train/"
log_path="./logs_review/"
log_filename="pos_analysis.log"
encoding=utf-8-sig

python ./src_review/do_posTagging.py --fp=$fp --log_fp=$log_path --log_filename=$log_filename --encoding=$encoding