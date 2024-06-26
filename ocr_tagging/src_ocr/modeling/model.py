import torch.nn as nn
from torchcrf import CRF
import transformers


class ABSAModel(nn.Module):
    def __init__(self, config, num_aspect, num_aspect2, need_birnn=False, rnn_dim=128):
        super(ABSAModel, self).__init__()
        # Sentiment와 Aspect Category의 class 개수
        self.num_aspect, self.num_aspect2 =  num_aspect, num_aspect2
        self.need_birnn = need_birnn

        # 사전 학습된 BERT를 load (최종 모델은 klue-bert 사용)
        self.bert = transformers.BertModel.from_pretrained(config.init_model_path)

        # Dropout layer
        self.aspect_drop, self.aspect2_drop = nn.Dropout(config.aspect_drop_ratio), nn.Dropout(config.aspect_drop_ratio)

        # Sentiment 및 Aspect Category layer 차원 설정
        aspect_in_feature = config.aspect_in_feature

        # birnn layer 추가
        if need_birnn:
            self.aspect_birnn, self.aspect2_birnn = nn.LSTM(aspect_in_feature, rnn_dim, num_layers=1,
                                                            bidirectional=True, batch_first=True),\
                                                    nn.LSTM(aspect_in_feature, rnn_dim,
                                                            num_layers=1, bidirectional=True, batch_first=True)

            aspect_in_feature = rnn_dim * 2

        # Sentiment와 Aspect Category의 Linear Layer 구성
        self.hidden2asptag, self.hidden2asp2tag = nn.Linear(aspect_in_feature, self.num_aspect), \
                                                  nn.Linear(aspect_in_feature, self.num_aspect2)
        # Sentiment와 Aspect Category의 CRF Layer 구성
        self.asp_crf, self.asp2_crf = CRF(self.num_aspect, batch_first=True), \
                                      CRF(self.num_aspect2, batch_first=True)

    def forward(self, ids, mask=None, token_type_ids=None, target_aspect=None, target_aspect2=None):
        # 사전학습된 bert에 input을 feed
        model_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[0]

        # BI-RNN layer
        if self.need_birnn:
            aspect_output, _ = self.aspect_birnn(model_output)
            aspect2_output, _ = self.aspect2_birnn(model_output)
        else:
            aspect_output, aspect2_output = model_output, model_output

        # 과적합 방지를 위해 Sentiment와 Aspect Category Dropout 수행
        aspect_output, aspect2_output = self.aspect_drop(aspect_output), self.aspect2_drop(aspect2_output)

        # Linear Layer feeding
        aspect_emmisions, aspect2_emmisions = self.hidden2asptag(aspect_output), self.hidden2asp2tag(aspect2_output)

        # CRF Layer Decoding
        aspect, aspect2 = self.asp_crf.decode(aspect_emmisions), self.asp2_crf.decode(aspect2_emmisions)

        # loss 계산
        if target_aspect is not None and target_aspect2 is not None:
            aspect_loss = (-1 * self.asp_crf(aspect_emmisions, target_aspect, mask=mask.byte())) + \
                          (-1 * self.asp2_crf(aspect2_emmisions, target_aspect2, mask=mask.byte()))
            loss = (aspect_loss) / 2

            return loss,  aspect, aspect2
        else:
            return aspect, aspect2







