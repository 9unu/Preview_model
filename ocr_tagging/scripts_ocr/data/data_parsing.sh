#!/bin/bash

# 경로에 띄어쓰기가 있다면 공백을 제거할 것!
fp="./resources_ocr/data/"
save_p="./resources_ocr/parsing_data/"
val_ratio=0.1
test_ratio=0.1
encoding=utf-8-sig

python ./src_ocr/do_parsingData.py --fp=$fp --save_p=$save_p --val_ratio=$val_ratio --test_ratio=$test_ratio --encoding=$encoding