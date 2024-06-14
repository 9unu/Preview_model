import pandas as pd
import os
import json
import re
from collections import Counter
import transformers
import kss
from pykospacing import Spacing # 추가한 부분

# tokenizer = transformers.BertTokenizer.from_pretrained('klue/bert-base', do_lower_case=False)('^^')
# print(tokenizer)
pattern1 = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+") # 한글 자모음만 반복되면 삭제
pattern2 = re.compile(r":\)|[\@\#\$\^\*\(\)\[\]\{\}\<\>\/\"\'\=\+\\\|\_(:\))]+") # ~, !, %, &, -, ,, ., ;, :, ?는 제거 X /// 특수문자 제거
pattern3 = re.compile(r"([^\d])\1{2,}") # 숫자를 제외한 동일한 문자 3개 이상이면 삭제
pattern4 = re.compile( # 이모티콘 삭제
    "["                               
    "\U0001F600-\U0001F64F"  # 감정 관련 이모티콘
    "\U0001F300-\U0001F5FF"  # 기호 및 픽토그램
    "\U0001F680-\U0001F6FF"  # 교통 및 지도 기호
    "\U0001F1E0-\U0001F1FF"  # 국기
    # "\U00002702-\U000027B0"  # 기타 기호
    # "\U000024C2-\U0001F251"  # 추가 기호 및 픽토그램      # 이거 2줄까지 하면 한글이 사라짐
    "]+", flags=re.UNICODE)


def regexp(sentences):
    for i in range(len(sentences)):
        sent = sentences[i]
        # og_sent = sent
        # if '"' in og_sent:
        #     print(og_sent)  
        new_sent1 = pattern1.sub('', sent)
        new_sent2 = pattern2.sub('', new_sent1)
        new_sent3 = pattern3.sub('', new_sent2)
        new_sent4 = pattern4.sub(r'', new_sent3)
        # if (og_sent != new_sent1 
        #     or og_sent != new_sent2 
        #     or og_sent != new_sent3
        #     ):
        #     print(f"og: {og_sent}, new: {new_sent1}")
        #     print(f"og: {og_sent}, new: {new_sent2}")
        #     print(f"og: {og_sent}, new: {new_sent3}")

        sentences[i] = new_sent4

    return sentences

def making_result_fp(args, filename):
    result_dir = args.save_p
    os.makedirs(result_dir, exist_ok=True)
    
    filename, ext = os.path.splitext(filename)
    result_fp = os.path.join(result_dir, f"{filename}.csv")
    
    return result_fp

def preprocess_text(text):
    return text.replace('\n', ' ')

def split_content_into_sentences(content): # 이 함수에서 정규표현식으로 특수문자 처리
    sentences = kss.split_sentences(content)
    sentences = regexp(sentences) # 수정한 부분
    return [preprocess_text(sent.strip()) + '.' for sent in sentences if sent.strip()]

def pykospaincg_preprocessing(sentences):  # 수정한 부분
    spacing = Spacing()
    for i in range(len(sentences)):
        sent = sentences[i]
        sent_spacingx = sent.replace(' ','') # 띄어쓰기 없애고
        pykospacing_sent = spacing(sent_spacingx) # pyko 돌리기
        sentences[i] = pykospacing_sent
        
    return sentences

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

def clean_data(our_topics):
    if not our_topics:
        return []
    
    cleansed_topics = []
    for topic in our_topics:
        if (not topic.get('text')
            or not topic.get("topic")
            or not topic.get("start_pos")
            or not topic.get("end_pos")
            or not topic.get("positive_yn")
            or not topic.get("sentiment_scale")
            or not topic.get("topic_score")
            ):
            continue

        cleansed_topics.append(topic)
    
    return cleansed_topics

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    
    rows = []
    review_counter = 1
    for item in data:
        if 'our_topics' not in item or not item['our_topics'] or 'content' not in item:            
            continue
        
        content = preprocess_text(item['content'])
        sentences = split_content_into_sentences(content)

        sentences = pykospaincg_preprocessing(sentences) # 추가한 부분(pyko)
        
        
        #  Add data cleansing about our_topics
        our_topics = clean_data(item['our_topics'])
        our_topics = sorted(our_topics, key=lambda x: len(x['text']), reverse=True)
        
        if not our_topics:
            continue
        
        sentence_counter = 1  # 문장 번호 초기화
        for sentence in sentences:
            words = sentence.split()
            tags = tag_sentence(sentence, our_topics)
            for word, tag in zip(words, tags):
                tag_parts = tag.split(',')
                sentiment = tag_parts[0] if len(tag_parts) > 0 else 'O'
                aspect = tag_parts[1] if len(tag_parts) > 1 else 'O'
                sentiment_Score = tag_parts[2] if len(tag_parts) > 2 else 'O'
                aspect_score = tag_parts[3] if len(tag_parts) > 3 else 'O'
                rows.append([f"Review {review_counter}", f"Sentence {sentence_counter}", word, sentiment, aspect, sentiment_Score, aspect_score])
            sentence_counter += 1  # 문장 번호 증가
        review_counter += 1
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows, columns=['Review #', 'Sentence #', 'Word', 'Sentiment', 'Aspect', 'Sentiment_Score', 'Aspect_Score'])
    return df

def process_json_files_in_folder(now_path, result_path):    
    json_file_path = now_path
    output_csv_path = result_path

    df = process_json_file(json_file_path)
    if df is not None:
        df.to_csv(output_csv_path, index=False)
        print(f"Processed and saved as {output_csv_path}")
    else:
        print(f"Skipping {json_file_path} due to no valid tagging data")


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

