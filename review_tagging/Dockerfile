FROM python:3.8.0

LABEL version="0.1.0"

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

