import torch.nn as nn
from torchcrf import CRF
import transformers

class ABSAModel(nn.Module):
    def __init__(self, config, num_sentiment, num_aspect, num_aspect2, num_sentiment_score, num_aspect_score, need_birnn=False, rnn_dim=128):
        super(ABSAModel, self).__init__()
        # Sentiment와 Aspect Category의 class 개수
        """옵션 불리언 초기화"""
        self.sentiment_score_bool = config.sentiment_score_bool
        self.aspect_score_bool = config.aspect_score_bool
        self.aspect_2_bool = config.aspect_2_bool
        self.num_sentiment, self.num_aspect = num_sentiment, num_aspect
        
        if self.sentiment_score_bool:
            self.num_sentiment_score = num_sentiment_score
        if self.aspect_score_bool:
            self.num_aspect_score = num_aspect_score 
        if self.aspect_2_bool:
            self.num_aspect2 = num_aspect2

        self.need_birnn = need_birnn

        # 사전 학습된 BERT를 load (최종 모델은 klue-bert 사용)
        self.bert = transformers.BertModel.from_pretrained(config.init_model_path)

        # Dropout layer
        self.sentiment_drop = nn.Dropout(config.sentiment_drop_ratio)
        self.aspect_drop = nn.Dropout(config.aspect_drop_ratio)  # sentiment랑 aspect만 dropout레이어 추가

        # Sentiment 및 Aspect Category layer 차원 설정
        sentiment_in_feature = config.sentiment_in_feature
        aspect_in_feature = config.aspect_in_feature
        
        # birnn layer 추가 설정 안함 (필요시 해야함)
        if need_birnn:
            self.sentiment_birnn = nn.LSTM(sentiment_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.sentiment_score_birnn = nn.LSTM(sentiment_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)  # 스코어 용
            self.aspect_score_birnn = nn.LSTM(aspect_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)  # 스코어 용
            
            self.aspect_birnn = nn.LSTM(aspect_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.aspect2_birnn = nn.LSTM(aspect_in_feature, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            sentiment_in_feature = rnn_dim * 2
            aspect_in_feature = rnn_dim * 2

        self.hidden2senttag = nn.Linear(sentiment_in_feature, self.num_sentiment) # B, I, 긍정 부정, PAD, O
        self.hidden2asptag = nn.Linear(aspect_in_feature, self.num_aspect) # B, I, PAD, O

        """옵션에 맞춰서 레이어 설정"""
        if self.sentiment_score_bool:
            self.hidden2sentscore = nn.Linear(sentiment_in_feature, self.num_sentiment_score)  # 스코어 레이어 추가
        if self.aspect_score_bool:
            self.hidden2aspectscore = nn.Linear(aspect_in_feature, self.num_aspect_score)  # 스코어 레이어 추가
        if self.aspect_2_bool:
            self.hidden2asp2tag = nn.Linear(aspect_in_feature, self.num_aspect2)    
        
        # Sentiment와 Aspect Category의 CRF Layer 구성
        self.sent_crf = CRF(self.num_sentiment, batch_first=True)
        self.asp_crf = CRF(self.num_aspect, batch_first=True)
        if self.sentiment_score_bool:
            self.sent_score_crf = CRF(self.num_sentiment_score, batch_first=True)  # 스코어 레이어 추가
        if self.aspect_score_bool:
            self.aspect_score_crf = CRF(self.num_aspect_score, batch_first=True)  # 스코어 레이어 추가
        if self.aspect_2_bool:
            self.asp2_crf = CRF(self.num_aspect2, batch_first=True) 

    def forward(self, ids, mask=None, token_type_ids=None, target_aspect=None, target_aspect2=None, target_sentiment=None, target_sentiment_score=None, target_aspect_score=None):
        # 사전학습된 bert에 input을 feed
        model_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[0]
        
        # BI-RNN layer
        if self.need_birnn:
            sentiment_output, _ = self.sentiment_birnn(model_output)
            sentiment_score_output, _ = self.sentiment_score_birnn(model_output)  # 스코어 레이어 추가
            aspect_score_output, _ = self.aspect_score_birnn(model_output)  # 스코어 레이어 추가
            aspect_output, _ = self.aspect_birnn(model_output)
            aspect2_output, _ = self.aspect2_birnn(model_output)
        else:
            sentiment_output = model_output
            sentiment_score_output = model_output
            aspect_score_output = model_output
            aspect_output = model_output
            aspect2_output = model_output  # 스코어 아웃풋 두개 추가

        # 과적합 방지를 위해 Sentiment와 Aspect Category Dropout 수행
        sentiment_output = self.sentiment_drop(sentiment_output)
        aspect_output = self.aspect_drop(aspect_output)
        
        # Linear Layer feeding
        sentiment_emmisions = self.hidden2senttag(sentiment_output)
        aspect_emmisions = self.hidden2asptag(aspect_output)
        if self.sentiment_score_bool:
            sentiment_score_emmisions = self.hidden2sentscore(sentiment_score_output)  # 스코어 아웃풋 두개 추가
        if self.aspect_score_bool:
            aspect_score_emmisions = self.hidden2aspectscore(aspect_score_output)  # 스코어 아웃풋 두개 추가
        if self.aspect_2_bool:
            aspect2_emmisions = self.hidden2asp2tag(aspect2_output)

        # CRF Layer Decoding
        sentiment = self.sent_crf.decode(sentiment_emmisions)
        aspect = self.asp_crf.decode(aspect_emmisions)
        if self.sentiment_score_bool:
            sentiment_score = self.sent_score_crf.decode(sentiment_score_emmisions)
        if self.aspect_score_bool:
            aspect_score = self.aspect_score_crf.decode(aspect_score_emmisions)
        if self.aspect_2_bool:
            aspect2 = self.asp2_crf.decode(aspect2_emmisions)
        
        # loss 계산
        loss_list = []
        target_list = []
        if target_aspect is not None and target_sentiment is not None:
            sentiment_loss = -1 * self.sent_crf(sentiment_emmisions, target_sentiment, mask=mask.byte())
            aspect_loss = -1 * self.asp_crf(aspect_emmisions, target_aspect, mask=mask.byte())
            loss_list.append(sentiment_loss)
            loss_list.append(aspect_loss)
            target_list.append(target_aspect)
            target_list.append(target_sentiment)

        if self.sentiment_score_bool and target_sentiment_score is not None:
            sentiment_score_loss = -1 * self.sent_score_crf(sentiment_score_emmisions, target_sentiment_score, mask=mask.byte())
            loss_list.append(sentiment_score_loss)
            target_list.append(target_sentiment_score)
        if self.aspect_score_bool and target_aspect_score is not None:
            aspect_score_loss = -1 * self.aspect_score_crf(aspect_score_emmisions, target_aspect_score, mask=mask.byte())
            loss_list.append(aspect_score_loss)
            target_list.append(target_aspect_score)
        if self.aspect_2_bool and target_aspect2 is not None:
            aspect_2_loss = -1 * self.asp2_crf(aspect2_emmisions, target_aspect2, mask=mask.byte())
            loss_list.append(aspect_2_loss)
            target_list.append(target_aspect2)
        
        loss = sum(loss_list) / len(loss_list) if loss_list else None

        if self.aspect_score_bool and self.sentiment_score_bool:
            if self.aspect_2_bool:
                for target in target_list:
                    if target is None:
                        return sentiment, aspect, aspect2, sentiment_score, aspect_score
                    else:
                        return loss, sentiment, aspect, aspect2, sentiment_score, aspect_score
            else:
                for target in target_list:
                    if target is None:
                        return sentiment, aspect, sentiment_score, aspect_score
                    else:
                        return loss, sentiment, aspect, sentiment_score, aspect_score
        
        for target in target_list:
            if target is None:
                return sentiment, aspect
            else:
                return loss, sentiment, aspect
