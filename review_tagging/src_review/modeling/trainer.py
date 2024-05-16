from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import torch
import time
import datetime


# input tensor의 구조 변경을 위한 함수
def parsing_batch(data, device):
    d = {}
    for k in data[0].keys():
        d[k] = list(d[k] for d in data)
    for k in d.keys():
        d[k] = torch.stack(d[k]).to(device)
    return d


# 모델 학습
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    loader_len = data_loader.dataset.get_length()
    for data in tqdm(data_loader, total=loader_len):
        data = parsing_batch(data, device)
        optimizer.zero_grad() # backward를 위한 gradients 초기화
        loss, sentiment, aspect, aspect2, sentiment_score, aspect_score = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / loader_len


def eval_fn(data_loader, model, enc_sentiment, enc_aspect, enc_sentiment_score, enc_aspect_score, device, log, f1_mode='micro', flag='valid'):
    model.eval()
    final_loss = 0
    nb_eval_steps = 0
    # 성능 측정 변수 선언
    sentiment_accuracy, aspect_accuracy = 0, 0
    sentiment_score_accuracy, aspect_score_accuracy = 0, 0  # 스코어 두개 정확도 지표

    sentiment_f1score, aspect_f1score = 0, 0
    sentiment_score_f1score, aspect_score_f1score = 0, 0  # 스코어 두개 f1 스코어 지표

    sentiment_preds, sentiment_labels = [], []
    sentiment_score_preds, sentiment_score_labels = [], []  # 스코어 두개 예측값, 라벨링 지표
    aspect_score_preds, aspect_score_labels = [], []  # 스코어 두개 예측값, 라벨링 지표

    aspect_preds, aspect_labels = [], []

    loader_len = data_loader.dataset.get_length()

    eval_start_time = time.time()  # evaluation을 시작한 시간을 저장 (소요 시간 측정을 위함)
    for data in tqdm(data_loader, total=loader_len):
        data = parsing_batch(data, device)
        loss, predict_sentiment, _, predict_aspect, predict_sentiment_score, predict_aspect_score = model(**data)
        sentiment_label = data['target_sentiment'].cpu().numpy().reshape(-1)
        aspect_label = data['target_aspect'].cpu().numpy().reshape(-1)
        sentiment_score_label = data['target_sentiment_score'].cpu().numpy().reshape(-1)
        aspect_score_label = data['target_aspect_score'].cpu().numpy().reshape(-1)

        sentiment_pred = np.array(predict_sentiment).reshape(-1)
        aspect_pred = np.array(predict_aspect).reshape(-1)
        sentiment_score_pred = np.array(predict_sentiment_score).reshape(-1)
        aspect_score_pred = np.array(predict_aspect_score).reshape(-1)

        # remove padding indices
        pad_label_indices = np.where(sentiment_label == 0)  # pad 레이블
        sentiment_label = np.delete(sentiment_label, pad_label_indices)
        sentiment_pred = np.delete(sentiment_pred, pad_label_indices)
        sentiment_score_label = np.delete(sentiment_score_label, pad_label_indices)
        sentiment_score_pred = np.delete(sentiment_score_pred, pad_label_indices)
        pad_label_indices = np.where(aspect_label == 0)  # pad 레이블
        aspect_label = np.delete(aspect_label, pad_label_indices)
        aspect_pred = np.delete(aspect_pred, pad_label_indices)
        aspect_score_label = np.delete(aspect_score_label, pad_label_indices)
        aspect_score_pred = np.delete(aspect_score_pred, pad_label_indices)

        # Accuracy 및 F1-score 계산
        sentiment_accuracy += accuracy_score(sentiment_label, sentiment_pred)
        aspect_accuracy += accuracy_score(aspect_label, aspect_pred)
        sentiment_f1score += f1_score(sentiment_label, sentiment_pred, average=f1_mode)
        aspect_f1score += f1_score(aspect_label, aspect_pred, average=f1_mode)

        sentiment_score_accuracy += accuracy_score(sentiment_score_label, sentiment_score_pred)
        aspect_score_accuracy += accuracy_score(aspect_score_label, aspect_score_pred)
        sentiment_score_f1score += f1_score(sentiment_score_label, sentiment_score_pred, average=f1_mode)
        aspect_score_f1score += f1_score(aspect_score_label, aspect_score_pred, average=f1_mode)

        # target label과 모델의 예측 결과를 저장 => classification report 계산 위함
        sentiment_labels.extend(sentiment_label)
        sentiment_preds.extend(sentiment_pred)
        sentiment_score_labels.extend(sentiment_score_label)
        sentiment_score_preds.extend(sentiment_score_pred)
        aspect_preds.extend(aspect_pred)
        aspect_labels.extend(aspect_label)
        aspect_score_preds.extend(aspect_score_pred)
        aspect_score_labels.extend(aspect_score_label)

        final_loss += loss.item()
        nb_eval_steps += 1

    # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)
    sentiment_pred_names = enc_sentiment.inverse_transform(sentiment_preds)
    sentiment_label_names = enc_sentiment.inverse_transform(sentiment_labels)
    
    sentiment_score_pred_names = enc_sentiment_score.inverse_transform(sentiment_score_preds)   ### score 예측값
    sentiment_score_label_names = enc_sentiment_score.inverse_transform(sentiment_score_labels) ### score 예측값 레이블

    aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
    aspect_label_names = enc_aspect.inverse_transform(aspect_labels)
    aspect_score_pred_names = enc_aspect_score.inverse_transform(aspect_score_preds)
    aspect_score_label_names = enc_aspect_score.inverse_transform(aspect_score_labels)

    # [Sentiment에 대한 성능 계산]
    sentiment_accuracy = round(sentiment_accuracy / nb_eval_steps, 2)
    sentiment_f1score = round(sentiment_f1score / nb_eval_steps, 2)
    
    sentiment_score_accuracy = round(sentiment_score_accuracy / nb_eval_steps, 2 )### score 예측값
    sentiment_score_f1score = round(sentiment_score_f1score / nb_eval_steps, 2) ### score 예측값
    
    # 각 감정 속성 별 성능 계산
    sentiment_report = classification_report(sentiment_label_names, sentiment_pred_names, digits=4)
    sentiment_score_report = classification_report(sentiment_score_label_names, sentiment_score_pred_names, digits=4)### score 예측값

    # [Aspect Category에 대한 성능 계산]
    aspect_accuracy = round(aspect_accuracy / nb_eval_steps, 2)
    aspect_f1score = round(aspect_f1score / nb_eval_steps, 2)
    aspect_score_accuracy = round(aspect_score_accuracy / nb_eval_steps, 2) ### score 예측값
    aspect_score_f1score = round(aspect_score_f1score / nb_eval_steps, 2) ### score 예측값
    
    # 각 aspect category에 대한 성능을 계산 (대분류 속성 기준)
    asp_report = classification_report(aspect_label_names, aspect_pred_names, digits=4)
    asp_score_report = classification_report(aspect_score_label_names, aspect_score_pred_names, digits=4) ### score 예측값

    eval_loss = final_loss / loader_len  # model의 loss
    eval_end_time = time.time() - eval_start_time  # 모든 데이터에 대한 평가 소요 시간
    eval_sample_per_sec = str(datetime.timedelta(seconds=(eval_end_time / loader_len)))  # 한 샘플에 대한 평가 소요 시간
    eval_times = str(datetime.timedelta(seconds=eval_end_time))  # 시:분:초 형식으로 변환

    # validation 과정일 때는, 각 sample에 대한 개별 결과값은 출력하지 않음
    if flag == 'eval':
        for i in range(len(aspect_label_names)):
            if aspect_label_names[i] != aspect_pred_names[i]:
                asp_result = "X"
            else:
                asp_result = "O"
            if sentiment_label_names[i] != sentiment_pred_names[i]:
                pol_result = "X"
            else:
                pol_result = "O"
            if sentiment_score_label_names[i] != sentiment_score_pred_names[i]:
                pol_score_result = "X"
            else:
                pol_score_result = "O"
            if aspect_score_label_names[i] != aspect_score_pred_names[i]:
                asp_score_result = "X"
            else:
                asp_score_result = "O"
            log.info(f"[{i} >> Sentiment : {pol_result} | Sentiment Score : {pol_score_result} | Aspect : {asp_result} | Aspect Score : {asp_score_result}] "
                     f"predicted sentiment label: {sentiment_pred_names[i]}, gold sentiment label: {sentiment_label_names[i]} | "
                     f"predicted sentiment score: {sentiment_score_pred_names[i]}, gold sentiment score: {sentiment_score_label_names[i]} | "
                     f"predicted aspect label: {aspect_pred_names[i]}, gold aspect label: {aspect_label_names[i]} | "
                     f"predicted aspect score: {aspect_score_pred_names[i]}, gold aspect score: {aspect_score_label_names[i]}")

    log.info("*****" + "eval metrics" + "*****")
    log.info(f"eval_loss: {eval_loss}")
    log.info(f"eval_runtime: {eval_times}")
    log.info(f"eval_samples: {loader_len}")
    log.info(f"eval_samples_per_second: {eval_sample_per_sec}")
    log.info(f"Sentiment Accuracy: {sentiment_accuracy}")
    log.info(f"Sentiment f1score {f1_mode} : {sentiment_f1score}")
    log.info(f"Sentiment Score Accuracy: {sentiment_score_accuracy}")
    log.info(f"Sentiment Score f1score {f1_mode} : {sentiment_score_f1score}")
    log.info(f"Aspect Accuracy: {aspect_accuracy}")
    log.info(f"Aspect f1score {f1_mode} : {aspect_f1score}")
    log.info(f"Aspect Score Accuracy: {aspect_score_accuracy}")
    log.info(f"Aspect Score f1score {f1_mode} : {aspect_score_f1score}")

    log.info(f"Sentiment Accuracy Report:")
    log.info(sentiment_report)
    log.info(f"Sentiment Score Accuracy Report:")
    log.info(sentiment_score_report)
    log.info(f"Aspect Accuracy Report:")
    log.info(asp_report)
    log.info(f"Aspect Score Accuracy Report:")
    log.info(asp_score_report)

    return eval_loss