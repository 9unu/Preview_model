import os
import re
import json
import pandas as pd
import kss

more_than_one_space = re.compile(r'\s{2,}')

def preprocess_text(text):
    return text.replace('\n', ' ').strip()

def split_content_into_sentences(content):
    sentences = kss.split_sentences(content)
    return [preprocess_text(sent.strip()) + "." for sent in sentences if sent.strip()]

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


def process_json_file(file_path, output_csv_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    
    rows = []
    review_counter = 1

    for item in data:
        content = more_than_one_space.sub(" ", preprocess_text(item['content']))
        sentences = split_content_into_sentences(content)
        
        sentence_dict_list = []        
        word_index = 0
        word_order = 0
        word_idx_to_sentences_mappping = {}

        for sent_num, sent in enumerate(sentences):
            for word in sent.split():                                                
                sentence_dict = {
                    'Review #': f"Review {review_counter}",
                    'Sentence #': f"Sentence {sent_num+1}", 
                    'Word': word, 
                    'Sentiment': 'O', 
                    'Aspect': 'O', 
                    'Sentiment_score': 'O', 
                    'Aspect_score': 'O',
                    }
                word_idx_to_sentences_mappping[word_index] = word_order   
                # 아래 부분 문제 가능성 높음.
                word_index += len(word) if word[-1] == "." else len(word)+1 # +1 for space
                word_order += 1
                sentence_dict_list.append(sentence_dict)
        # 0~5가 단어. sentence_dict_list의 0 index가 단어를 표현하는 거에요.
        # 7~10가 단어. sentence_dict_list의 1 index가 해당하는 단어인거에요.
        # 12~13가 단어. sentence_dict_list의 2 index가 해당하는 단어인거에요.

        # .이 있다. 그러면 12~14까지가 단어지만 sentence_dict_list의 3 index가 해당하는 단어인거에요.

        sentence_words_concat = content                                              

        our_topics = sorted(item['our_topics'], key=lambda x: len(x['text']), reverse=True)
        checked_indice_set = set()

        for topic in our_topics:
            topic_text = more_than_one_space.sub(" ", preprocess_text(topic['text']))

            start_idx = sentence_words_concat.find(topic_text)
            # 이미 체크한 인덱스인지 확인
            while start_idx != -1 and start_idx in checked_indice_set:
                # 체크한 인덱스라면 다음 인덱스부터 찾기
                start_idx = sentence_words_concat.find(topic_text, start_idx+1)
            # 토픽이 발견되지 않았을 때 (이 경우 에러임)
            if start_idx == -1:
                # raise ValueError(f"Topic '{topic_text}' not found in review {review_counter}")            
                print(f"Topic '{topic_text}' not found or duplicate in review {review_counter} at {file_path}")
                continue
            
            # 처음 발견된 토픽이었을 때                        
            end_idx = start_idx + len(topic_text) - len(topic_text.split()[-1]) # 마지막 단어의 시작 인덱스 찾기

            # 체크 인덱스에 추가
            for i in range(start_idx, start_idx+len(topic_text)):
                checked_indice_set.add(i)

            start_word_idx = word_idx_to_sentences_mappping[start_idx]
            end_word_idx = word_idx_to_sentences_mappping[end_idx]
            sentence_dict_list[start_word_idx]['Sentiment'] = f"{'B-긍정' if topic['positive_yn'] == 'Y' else 'B-부정'}"
            sentence_dict_list[start_word_idx]['Aspect'] = f"B-{topic['topic']}"
            sentence_dict_list[start_word_idx]['Sentiment_score'] = f"B-{topic['sentiment_scale']}"
            sentence_dict_list[start_word_idx]['Aspect_score'] = f"B-{topic['topic_score']}"                
            for word_idx in range(start_word_idx+1, end_word_idx+1):
                sentence_dict_list[word_idx]['Sentiment'] = f"{'I-긍정' if topic['positive_yn'] == 'Y' else 'I-부정'}"
                sentence_dict_list[word_idx]['Aspect'] = f"I-{topic['topic']}"
                sentence_dict_list[word_idx]['Sentiment_score'] = f"I-{topic['sentiment_scale']}"
                sentence_dict_list[word_idx]['Aspect_score'] = f"I-{topic['topic_score']}"
        
        rows.extend(sentence_dict_list)
        review_counter += 1

        # 이것들을 계속 이어줘야 함. 디버깅 필요.
        #  df = pd.DataFrame(sentence_dict_list)         
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    else:
        print(f"Skipping {file_path} due to no valid tagging data")

    return
    

def process_json_files_in_folder(folder_path, output_folder):
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 폴더 내의 모든 JSON 파일 처리
    for filename in os.listdir(folder_path):
        if filename != '20240401_06h14m38s_extra_battery_벨킨 BPD004bt_review.json':
            continue
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)
            output_csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
            
            # JSON 파일 처리 및 CSV 파일 생성
            process_json_file(json_file_path, output_csv_path)
            
            print(f"Processed {filename} and saved as {output_csv_path}")

# 폴더 경로 설정
json_folder_path = r"C:\Users\82104\Python_workspace\SFTP\Preview_model\review_tagging\resources_review\data"
output_folder_path = r"C:\Users\82104\Python_workspace\SFTP\Preview_model\review_tagging\resources_review\data\parsed_csv"

# 폴더 내의 모든 JSON 파일 처리 및 CSV 파일 생성
process_json_files_in_folder(json_folder_path, output_folder_path) 
