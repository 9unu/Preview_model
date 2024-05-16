import pandas as pd
import os
import json
import re
from collections import Counter
import kss


def making_result_fp(args, filename):
    result_dir = args.save_p
    os.makedirs(result_dir, exist_ok=True)
    
    filename, ext = os.path.splitext(filename)
    result_fp = os.path.join(result_dir, f"{filename}.csv")
    
    return result_fp

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
        if 'our_topics' not in item or not item['our_topics']:
            continue
        
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
                            sentiment_Score = tag_parts[2] if len(tag_parts) > 2 else '0'
                            aspect_score = tag_parts[3] if len(tag_parts) > 3 else '0'
                            rows.append([f"Sentence {sentence_counter}", word, sentiment, aspect, sentiment_Score, aspect_score])
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
                    sentiment_Score = tag_parts[2] if len(tag_parts) > 2 else '0'
                    aspect_score = tag_parts[3] if len(tag_parts) > 3 else '0'
                    rows.append([f"Sentence {sentence_counter}", word, sentiment, aspect, sentiment_Score, aspect_score])
                sentence_counter += 1
                sent_idx += 1
    
    df = pd.DataFrame(rows, columns=['Sentence #', 'Word', 'Sentiment', 'Aspect', 'Sentiment_Score', 'Aspect_Score'])
    return df

def process_json_files_in_folder(now_path, result_path):    
    json_file_path = now_path
    output_csv_path = result_path

    df = process_json_file(json_file_path)
    df.to_csv(output_csv_path, index=False)
    print(f"Processed and saved as {output_csv_path}")


def json_2_csv(args):
    json_list=os.listdir(args.fp)
    result_path=[]
    now_path=[]
    for file_name in json_list:
        if file_name.endswith(".json"):
            now_path.append(os.path.join(args.fp, file_name))
            result_fp = making_result_fp(args, file_name)
            result_path.append(result_fp)
    
    for a, b in zip(now_path, result_path):
        process_json_files_in_folder(a, b)

       