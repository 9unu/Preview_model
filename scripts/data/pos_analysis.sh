#!/bin/bash

fp="./resources/data/train/"
log_path="./logs/"
log_filename="pos_analysis.log"
encoding=utf-8-sig

python ./src/do_posTagging.py --fp=$fp --log_fp=$log_path --log_filename=$log_filename --encoding=$encoding