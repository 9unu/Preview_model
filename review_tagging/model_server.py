from http.server import BaseHTTPRequestHandler
import argparse
import json
import os
from google.cloud import storage
import joblib
import transformers
from fastapi import FastAPI, HTTPException, APIRouter, Request
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager

from .src_review.modeling.model import ABSAModel
from .src_review.modeling.trainer import tag_fn
from .src_review.utils.model_utils import device_setting, load_model
from .src_review.utils.set_logger import Log


class ServeLog(Log):    
    def __init__(self):
        super().__init__()
        
    def set_stream_handler(self):
        super().set_stream_handler()
    
    def set_log(self, level="DEBUG"):
        self.set_stream_handler()
        self.log.setLevel(self.levels[level])
        return self.log

model = None
log = ServeLog().set_log(level="DEBUG") # only stream.
tokenizer = None

def load_bert_model(config: argparse.Namespace):
    global log
    log.info(f"Loading BERT model. Config: {config.__dict__}")
    
    bucket_name = config.base_path
    model_file = config.out_model_path
    metadata_file = config.label_info_file
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    model_blob = bucket.blob(model_file)
    metadata_blob = bucket.blob(metadata_file)        
    
    local_model_path = '/tmp/pytorch_model.bin'
    metadata_path = '/tmp/meta.bin'
    
    model_blob.download_to_filename(local_model_path)        
    log.info(f"Model downloaded to {local_model_path}")
    metadata_blob.download_to_filename(metadata_path)
    log.info(f"Metadata (label) downloaded to {metadata_path}")
    
    metadata = joblib.load(metadata_path)
    enc_sentiment, enc_aspect, enc_aspect2 = metadata["enc_sentiment"], metadata["enc_aspect"], metadata["enc_aspect2"]
    enc_sentiment_score, enc_aspect_score = metadata["enc_sentiment_score"], metadata["enc_aspect_score"]
    
    num_sentiment = len(list(metadata["enc_sentiment"].classes_))
    num_aspect, num_aspect2 = len(list(metadata["enc_aspect"].classes_)), len(list(metadata["enc_aspect2"].classes_))
    num_sentiment_score = len(list(metadata["enc_sentiment_score"].classes_)) ### score_num 추가
    num_aspect_score = len(list(metadata["enc_aspect_score"].classes_)) ### score_num 추가
    
    device = device_setting(log)
    global model
    model = ABSAModel(config=config, num_sentiment=num_sentiment, num_aspect=num_aspect, num_aspect2=num_aspect2,
                            num_sentiment_score=num_sentiment_score, num_aspect_score=num_aspect_score,
                            need_birnn=bool(config.need_birnn))
    model = load_model(model=model, state_dict_path=local_model_path, device=device)
    global tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(config.init_model_path, do_lower_case=False)

def inference(text: str):
    global model
    global tokenizer
    global log
    
    log.info(f"Text: {text}")
    result = tag_fn(text, tokenizer, model, log)
    log.info(f"Result: {result}")
    return result

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.post("/predict")
async def predict():
    
    # 비동기 실행 컨텍스트 생성
    loop = asyncio.get_running_loop()
    
    # predict_fn을 비동기적으로 실행
    text = await loop.run_in_executor(None, predict_fn)
    
    # 후처리는 순차적으로 실행
    post_text = postprocess(text)
    
    text = predict_fn()
    post_text = postprocess(text)
    
    json_text["our_topics"] = post_text
    
    res = await reqeusts.post(url, method=["POST"], json_text)
    
    
    return res

def create_app(config: argparse.Namespace):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        load_bert_model(config)
        yield
        print("Cleaning up")        

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    
    return app


    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=1, help="한 batch에 속할 테스트 데이터 샘플의 size")
    parser.add_argument("--init_model_path", type=str, default="klue/bert-base", help="사용된 BERT의 종류")
    parser.add_argument("--max_length", type=int, default=512, help="토큰화된 문장의 최대 길이(bert는 기본 512)")
    parser.add_argument("--need_birnn", type=int, default=0, help="model에 Birnn Layer를 추가했는지 여부 (True: 1/False: 0)")
    parser.add_argument("--sentiment_drop_ratio", type=float, default=0.3,
                        help="Sentiment 속성의 과적합 방지를 위해 dropout을 수행한 비율")
    parser.add_argument("--aspect_drop_ratio", type=float, default=0.3,
                        help="Aspect Category 속성의 과적합 방지를 위해 dropout을 수행한 비율")
    parser.add_argument("--sentiment_in_feature", type=int, default=768,
                        help="각 Sentiment input sample의 size")
    parser.add_argument("--aspect_in_feature", type=int, default=768,
                        help="각 Aspect Category input sample의 size")
    parser.add_argument("--base_path", type=str, help="평가를 수행할 Model과 Encoder가 저장된 bucket 경로")
    ### 경로 수정
    parser.add_argument("--label_info_file", type=str, help="사용할 Encoder 파일명", default="meta.bin")
    parser.add_argument("--out_model_path", type=str, help="평가할 model의 파일명", default="pytorch_model.bin")
    
    config = parser.parse_args()
    
    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=8000)