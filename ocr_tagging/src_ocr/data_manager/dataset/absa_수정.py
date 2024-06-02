from pathlib import Path
import os
import joblib
from sklearn import preprocessing
from sklearn.utils import column_or_1d
from collections import OrderedDict
import pandas as pd
import transformers
import torch
from torch.utils.data import IterableDataset
from utils.file_io import get_file_list, read_csv
from data_manager.parsers.label_unification.label_map import label_list, label_changing_rule
import math


# Encoder - Sentiment 및 Aspect 속성을 Encoding할 Label Encoder class
class MyLabelEncoder(preprocessing.LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self


class Encoder:
    def __init__(self, config, fp=None, extension="csv"):
        self.fp = fp
        self.file_list = get_file_list(fp, extension)
        self.meta_data_fp = os.path.join(config.base_path + config.label_info_file)

        # Sentiment 속성을 위한 Encoder 및 Aspect Category를 위한 Encoder
        self.enc_aspect, self.enc_aspect2 = None, None
        self.aspect_labels, self.aspect2_labels = ['PAD', 'O'], ['PAD', 'O']

    # 저장된 Encoder 유무를 확인
    # 있으면 load, 없으면 생성
    def check_encoder_fp(self):
        meta_data_f = Path(self.meta_data_fp)
        if meta_data_f.exists():
            meta_data = joblib.load(self.meta_data_fp)
            self.enc_aspect, self.enc_aspect2 = meta_data["enc_aspect"], meta_data["enc_aspect2"]
        else:
            meta_data = self.set_encoder()
            joblib.dump(meta_data, self.meta_data_fp)

    def get_encoder(self):
        if self.enc_aspect is None or self.enc_aspect2 is None:
            self.check_encoder_fp()
        return self.enc_aspect, self.enc_aspect2

    def set_encoder(self):
        if len(self.file_list) == 0:
            print(f"파일 경로 {self.fp}에 Encoding할 데이터가 존재하지 않습니다.")
            raise FileExistsError()

        for now_fp in self.file_list:
            df = read_csv(now_fp)
            self.aspect_labels.extend(list(df["Aspect"].unique()))

        # Encoder 선언 및 fitting
        self.enc_aspect, self.enc_aspect2 = MyLabelEncoder(), MyLabelEncoder()
        self.aspect_labels = list(OrderedDict.fromkeys(self.aspect_labels))
        self.aspect2_labels.extend([label for label in label_list])

        self.enc_aspect = self.enc_aspect.fit(self.aspect_labels)
        self.enc_aspect2 = self.enc_aspect2.fit(self.aspect2_labels)
        return {"enc_aspect": self.enc_aspect, "enc_aspect2": self.enc_aspect2}


class ABSADataset(IterableDataset):
    def __init__(self, config, fp, enc_aspect, enc_aspect2, batch_size, data_len=0, extension="csv"):
        self.data_len = data_len
        self.file_list = get_file_list(fp, extension)
        self.batch_size = batch_size
        self.max_len = config.max_length
        self.config = config

        # Encoder Setting
        self.enc_aspect = enc_aspect
        self.enc_aspect2 = enc_aspect2

        # for embedding
        self.tokenizer = transformers.BertTokenizer.from_pretrained(config.init_model_path, do_lower_case=False)
        self.CLS_IDS = self.tokenizer.encode('[CLS]', add_special_tokens=False)  # [2]
        self.PAD_IDS = self.tokenizer.encode('[PAD]', add_special_tokens=False)  # [0]
        self.SEP_IDS = self.tokenizer.encode('[SEP]', add_special_tokens=False)  # [3]
        self.PADDING_TAG_IDS = [0]
        self.s_len = 0

    def __iter__(self):
        # read data
        for now_fp in self.file_list:
            df = read_csv(now_fp)
            df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
            df["Aspect2"] = df["Aspect"]
            df = df.replace({"Aspect2": label_changing_rule})

            df.loc[:, "Aspect"] = self.enc_aspect.transform(df[["Aspect"]])
            df.loc[:, "Aspect2"] = self.enc_aspect2.transform(df[["Aspect2"]])

            sentences = df.groupby("Sentence #")["Word"].apply(list).values
            aspects = df.groupby("Sentence #")["Aspect"].apply(list).values
            aspects2 = df.groupby("Sentence #")["Aspect2"].apply(list).values

            # for i in range(len(sentences)):
            #     self.s_len += 1
            #     yield self.parsing_data(sentences[i], aspects[i], aspects2[i])
            for i in range(len(sentences)):
                self.s_len += 1
                if(i % 2 == 0):
                    yield self.parsing_data(sentences[i], sentences[i+1], aspects[i], aspects[i+1], aspects2[i], aspects2[i+1])
                elif(i & 2 == 0 and i == len(sentences)-1):
                    yield self.parsing_data(sentences[i], sentences[i], aspects[i], aspects[i], aspects2[i], aspects2[i])
                # if(i == len(sentences)-1):
                #     self.s_len += 2
                #     yield self.parsing_data(sentences[i], sentences[i], aspects[i], aspects[i], aspects2[i], aspects2[i])
                # else:
                #     self.s_len += 2
                #     yield self.parsing_data(sentences[i], sentences[i+1], aspects[i], aspects[i+1], aspects2[i], aspects2[i+1])

    def __len__(self):
        if self.data_len == 0:
            self.data_len = self.get_length()
        return self.data_len

    # data length를 계산
    def get_length(self):
        if self.data_len > 0:
            return self.data_len
        else:
            for now_fp in self.file_list:
                df = read_csv(now_fp)
                sentences = df.groupby("Sentence #")["Word"].apply(list).values
                self.data_len += len(sentences)
            self.data_len = math.ceil(self.data_len / self.batch_size)
            return self.data_len
    
    def parsing_data(self, text_1, text_2, aspect_1, aspect_2, aspect2_1, aspect2_2):
        ids = []
        target_aspect = [] # target Aspect Category tensor ids 저장 리스트
        target_aspect2 = []  # target 대분류 Aspect Category tensor ids 저장 리스트 (대분류 기준 성능 측정을 위함)

        # for i, s in enumerate(text):
        #     inputs = self.tokenizer.encode(s, add_special_tokens=False)
        #     input_len = len(inputs)
        #     ids.extend(inputs)
        #     target_aspect.extend([aspect[i]] * input_len)
        #     target_aspect2.extend([aspect2[i]] * input_len)
        for i, (s1, s2) in enumerate(zip(text_1, text_2)):
            inputs_1 = self.tokenizer.encode(s1, add_special_tokens=False)
            inputs_2 = self.tokenizer.encode(s2, add_special_tokens=False)
            inputs_1_len = len(inputs_1)
            inputs_2_len = len(inputs_2)
            ids.extend(inputs_1 + self.SEP_IDS + inputs_2)
            target_aspect.extend([aspect_1[i]] * inputs_1_len + self.PADDING_TAG_IDS + [aspect_2[i]] * inputs_2_len)
            target_aspect2.extend([aspect2_1[i]] * inputs_1_len + self.PADDING_TAG_IDS + [aspect2_2[i]] * inputs_2_len)

        # BERT가 처리할 수 있는 길이 (max_length)에 맞추어 slicing
        ids = ids[:self.max_len - 2]
        target_aspect = target_aspect[:self.max_len - 2]
        target_aspect2 = target_aspect2[:self.max_len - 2]

        # SPECIAL TOKEN 추가 및 PADDING 수행
        ids = self.CLS_IDS + ids + self.SEP_IDS
        target_aspect = self.PADDING_TAG_IDS + target_aspect + self.PADDING_TAG_IDS  # CLS, SEP 태그 0
        target_aspect2 = self.PADDING_TAG_IDS + target_aspect2 + self.PADDING_TAG_IDS

        mask = [1] * len(ids)
        token_type_ids = self.PAD_IDS * len(ids)
        padding_len = self.max_len - len(ids)
        ids = ids + (self.PAD_IDS * padding_len)
        mask = mask + ([0] * padding_len)

        token_type_ids = token_type_ids + (self.PAD_IDS * padding_len)
        target_aspect = target_aspect + (self.PADDING_TAG_IDS * padding_len)
        target_aspect2 = target_aspect2 + (self.PADDING_TAG_IDS * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_aspect": torch.tensor(target_aspect, dtype=torch.long),
            "target_aspect2": torch.tensor(target_aspect2, dtype=torch.long),
            }

