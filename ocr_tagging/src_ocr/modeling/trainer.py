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
        loss, aspect, aspect2 = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / loader_len


# 모델 평가
def eval_fn(data_loader, model, enc_aspect, device, log, f1_mode='micro', flag='valid'):
    model.eval()
    final_loss = 0
    nb_eval_steps = 0
    # 성능 측정 변수 선언
    aspect_accuracy = 0
    aspect_f1score = 0
    aspect_preds, aspect_labels = [], []

    loader_len = data_loader.dataset.get_length()

    eval_start_time = time.time() # evaluation을 시작한 시간을 저장 (소요 시간 측정을 위함)
    for data in tqdm(data_loader, total=loader_len):
        data = parsing_batch(data, device)
        loss, _, predict_aspect = model(**data)
        aspect_label = data['target_aspect'].cpu().numpy().reshape(-1)

        aspect_pred = np.array(predict_aspect).reshape(-1)

        #remove padding indices
        pad_label_indices = np.where(aspect_label == 0)  # pad 레이블
        aspect_label = np.delete(aspect_label, pad_label_indices)
        aspect_pred = np.delete(aspect_pred, pad_label_indices)

        # Accuracy 및 F1-score 계산
        aspect_accuracy += accuracy_score(aspect_label, aspect_pred)
        aspect_f1score += f1_score(aspect_label, aspect_pred, average=f1_mode)

        # target label과 모델의 예측 결과를 저장 => classification report 계산 위함
        aspect_preds.extend(aspect_pred)
        aspect_labels.extend(aspect_label)

        final_loss += loss.item()
        nb_eval_steps += 1

    # encoding 된 Sentiment와 Aspect Category를 Decoding (원 형태로 복원)

    aspect_pred_names = enc_aspect.inverse_transform(aspect_preds)
    aspect_label_names = enc_aspect.inverse_transform(aspect_labels)
    
    # [Aspect Category에 대한 성능 계산]
    aspect_accuracy = round(aspect_accuracy / nb_eval_steps, 2)
    aspect_f1score = round(aspect_f1score / nb_eval_steps, 2)
    # 각 aspect category에 대한 성능을 계산 (대분류 속성 기준)
    asp_report = classification_report(aspect_label_names, aspect_pred_names, digits=4)
    
    eval_loss = final_loss / loader_len # model의 loss
    eval_end_time = time.time() - eval_start_time # 모든 데이터에 대한 평가 소요 시간
    eval_sample_per_sec = str(datetime.timedelta(seconds=(eval_end_time/loader_len)))# 한 샘플에 대한 평가 소요 시간
    eval_times = str(datetime.timedelta(seconds=eval_end_time)) # 시:분:초 형식으로 변환

    # validation 과정일 때는, 각 sample에 대한 개별 결과값은 출력하지 않음
    if flag == 'eval':
        for i in range(len(aspect_label_names)):
            if aspect_label_names[i] != aspect_pred_names[i]:
                asp_result = "X"
            else:
                asp_result = "O"
            log.info(f"[{i} >> Aspect : {asp_result}] predicted label:"
                     f"predicted aspect label: {aspect_pred_names[i]}, gold aspect label: {aspect_label_names[i]}")

    log.info("*****" + "eval metrics" + "*****")
    log.info(f"eval_loss: {eval_loss}")
    log.info(f"eval_runtime: {eval_times}")
    log.info(f"eval_samples: {loader_len}")
    log.info(f"eval_samples_per_second: {eval_sample_per_sec}")
    log.info(f"Aspect Accuracy: {aspect_accuracy}")
    log.info(f"Aspect f1score {f1_mode} : {aspect_f1score}")
    log.info(f"Aspect Accuracy Report:")
    log.info(asp_report)

    return eval_loss