REGION=asia-northeast3
PROJECT-ID=webserver-project-313910


docker build -t kluebert-review:v1 --no-cache -f .\scripts_review\deploy\Dockerfile .
docker tag kluebert-review:v1 [REGION]-docker.pkg.dev/[PROJECT-ID]/[REPOSITORY]/kluebert-review:v1
docker run -d -p 8000:8000 kluebert-review:v1 \
    --eval_batch_size 1 \
    --init_model_path klue/bert-base \
    --max_length 512 \
    --need_birnn 0 \
    --sentiment_drop_ratio 0.3 \
    --aspect_drop_ratio 0.3 \
    --sentiment_in_feature 768 \
    --aspect_in_feature 768 \
    --base_path review_tagging \
    --out_model_path meta.bin --label_info_file pytorch_model.bin \
    --post_server http://35.216.119.230/api/review


# 빌드 시에 환경변수를 override할 수 있음.
gcloud builds submit --config=cloudbuild.yaml --substitutions=_POST_SERVER=http://35.216.119.230/api/review

