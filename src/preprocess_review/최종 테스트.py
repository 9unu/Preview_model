import os
import re
import json
import pandas as pd
import kss

def preprocess_text(text):
    return text.replace('\n', ' ')

def split_content_into_sentences(content):
    sentences = kss.split_sentences(content)
    return [preprocess_text(sent.strip()) for sent in sentences if sent.strip()]

def tag_sentence(sentence, topics):
    words = sentence.split()
    tags = ['O'] * len(words)
    for topic in topics:
        topic_text = preprocess_text(topic['text'])
        topic_words = topic_text.split()
        start_idx = 0
        while True:
            idx = sentence.find(topic_text, start_idx)
            if idx == -1:
                break
            end_idx = idx + len(topic_text)
            word_idx = len(sentence[:idx].split())
            for j in range(word_idx, word_idx + len(topic_words)):
                if j >= len(words):
                    break
                if j == word_idx:
                    tags[j] = f"{'B-긍정' if topic['positive_yn'] == 'Y' else 'B-부정'},B-{topic['topic']},B-{topic['sentiment_scale']},B-{topic['topic_score']}"
                else:
                    tags[j] = f"{'I-긍정' if topic['positive_yn'] == 'Y' else 'I-부정'},I-{topic['topic']},I-{topic['sentiment_scale']},I-{topic['topic_score']}"
            start_idx = end_idx
    return tags

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    
    rows = []
    sentence_counter = 1
    for item in data:
        content = preprocess_text(item['content'])
        sentences = split_content_into_sentences(content)
        our_topics = sorted(item['our_topics'], key=lambda x: len(x['text']), reverse=True)
        
        sent_idx = 0
        while sent_idx < len(sentences):
            concat_sent = ""
            for sent_concat_count in range(3, 0, -1):
                if sent_idx + sent_concat_count > len(sentences):
                    continue
                concat_sent = " ".join(sentences[sent_idx:sent_idx+sent_concat_count])
                for topic in our_topics:
                    if preprocess_text(topic['text']) in concat_sent:
                        words = concat_sent.split()
                        tags = tag_sentence(concat_sent, our_topics)
                        for word, tag in zip(words, tags):
                            tag_parts = tag.split(',')
                            sentiment = tag_parts[0] if len(tag_parts) > 0 else 'O'
                            aspect = tag_parts[1] if len(tag_parts) > 1 else 'O'
                            sentiment_score = tag_parts[2] if len(tag_parts) > 2 else 'O'
                            aspect_score = tag_parts[3] if len(tag_parts) > 3 else 'O'
                            rows.append([f"Sentence{sentence_counter}", word, sentiment, aspect, sentiment_score, aspect_score])
                        sentence_counter += 1
                        sent_idx += sent_concat_count
                        break
                else:
                    continue
                break
            else:
                concat_sent = sentences[sent_idx]
                words = concat_sent.split()
                tags = tag_sentence(concat_sent, our_topics)
                for word, tag in zip(words, tags):
                    tag_parts = tag.split(',')
                    sentiment = tag_parts[0] if len(tag_parts) > 0 else 'O'
                    aspect = tag_parts[1] if len(tag_parts) > 1 else 'O'
                    sentiment_score = tag_parts[2] if len(tag_parts) > 2 else 'O'
                    aspect_score = tag_parts[3] if len(tag_parts) > 3 else 'O'
                    rows.append([f"Sentence{sentence_counter}", word, sentiment, aspect, sentiment_score, aspect_score])
                sentence_counter += 1
                sent_idx += 1
    
    df = pd.DataFrame(rows, columns=['Sentence#', 'Word', 'Sentiment', 'Aspect', 'Sentiment_score', 'Aspect_score'])
    return df

def process_json_files_in_folder(folder_path, output_folder):
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 폴더 내의 모든 JSON 파일 처리
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)
            output_csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
            
            # JSON 파일 처리 및 CSV 파일 생성
            df = process_json_file(json_file_path)
            df.to_csv(output_csv_path, index=False)
            print(f"Processed {filename} and saved as {output_csv_path}")

# 폴더 경로 설정
json_folder_path = r"C:\Users\dnwfr\Desktop\대학 관련\4학년 1학기\창의적종합설계\test"
output_folder_path = r"C:\Users\dnwfr\Desktop\output_folder"

# 폴더 내의 모든 JSON 파일 처리 및 CSV 파일 생성
process_json_files_in_folder(json_folder_path, output_folder_path)