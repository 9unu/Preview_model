import pandas as pd
import os
import json
import re
from collections import Counter

polarity_dict = {"-1": "부정",
                 "0": "중립",
                 "1": "긍정"}


def making_result_fp(args, dir, ext):
    result = os.path.join(args.save_p, dir.split(args.fp)[-1])
    if "." not in ext:
        result += "."
    return result + ext


def is_sublist(left, right):
    for i in range(len(right) - len(left) + 1):
        if right[i:i + len(left)] == left:
            return (i, i + len(left))
    return None


def preprocess_sentence(sentence):
    words = sentence.replace("\n", " ")
    words = words.replace(",", " ")  #
    words = words.replace(",", " ")  #
    words = re.sub(' +', ' ', words)  # to single white_space
    return words


def sentence2words(sentence):
    words = preprocess_sentence(sentence)
    words = words.split(" ")
    words = list(filter(None, words))
    return words


def check_bad_example(sentence, s_index, t_index):
    if sentence == 'nan' or type(sentence) != str:
        print("drop index {s_index}-".format(s_index=s_index + 2) + str(t_index))  # s_index +2
        return True
    return False


def printing_error_size(name, total_example, total_processed_example, example_nan_error, label_match_error, label_error,
                        polarity_dist_dict, aspect_dist_dict, save_p, encoding):
    print("\n" + '*' * 50)
    print("About {example}\t".format(example=name))
    print("<데이터 사이즈>")
    print("{example}\t total_example".format(example=total_example))
    print("{example}\t total_processed_example".format(example=total_processed_example))
    print("{error}\t example_nan_error(데이터셋에서 제외)".format(error=example_nan_error))  # 입력 문장이 None 으로 인식 되는 경우
    print("{error}\t label_match_error(레이블 태깅안함)".format(error=label_match_error))  # 입력 문장과 레이블 스팬이 일치 하지 않는 경우
    print("{error}\t label_error(레이블 태깅 안함)".format(error=label_error))

    print("\n" + '/' * 50)
    dist_info = {"감정": polarity_dist_dict, "속성": aspect_dist_dict}
    for feature, dist_dict in dist_info.items():
        print(f"<{feature} 레이블 분포>")
        dist_df = pd.DataFrame([dict(dist_dict)]).T
        dist_df.columns = ["분포"]
        print(dist_df)


def cal_distribution(dic, key):
    if key in dic:
        dic[key] += 1
    else:
        dic[key] = 1
    return dic


def remove_spacing(sentence):
    words = preprocess_sentence(sentence)
    words = words.replace(" ", "")
    return words


def json_2_csv(args, extension=".json", output_extension="csv"):
    fp = args.fp
    save_p = args.save_p

    dir_list = [os.path.join(fp, dir) for dir in os.listdir(fp) if os.path.isdir(os.path.join(fp, dir))]
    if not os.path.exists(save_p):
        os.makedirs(save_p)

    entire_total_example = 0 # 전체 example data
    entire_total_processed_example = 0 # 전체 데이터에서 처리된 data 수
    entire_example_nan_error = 0  # 전체 데이터에서 입력 문장이 None 으로 인식 되는 경우
    entire_label_match_error = 0  # 전체 데이터에서 입력 문장과 레이블 스팬이 일치 하지 않는 경우
    entire_label_error = 0  # 전체 데이터에서 label 이 잘못 표시 된 경우 (O가 0으로 표시된 경우)
    # 전체 데이터에서 각 속성 내 label의 분포를 계산하기 위한 class
    entire_aspect_dist_dict = Counter({})
    entire_polarity_dist_dict = Counter({})

    for dir in dir_list:
        print("\n" + '*' * 50)
        print("now parsing Directory: {name}".format(name=dir))
        # Per Directory
        total_example = 0 # 해당 디렉토리 내, 처리된 data 수
        example_nan_error = 0  # 입력 문장이 None 으로 인식 되는 경우
        label_match_error = 0  # 입력 문장과 레이블 스팬이 일치 하지 않는 경우
        label_error = 0 # label 이 잘못 표시 된 경우 (O가 0으로 표시된 경우)
        # 각 속성 내 label의 분포를 계산하기 위한 class
        aspect_dist_dict = Counter({})
        polarity_dist_dict = Counter({})

        # 결과를 저장할 파일명 생성
        result_fp = making_result_fp(args, dir, output_extension)

        # 현 디렉터리 내, 파싱할 파일 경로로 list를 구성
        file_name_list = [os.path.join(dir, file) for file in os.listdir(dir)]

        # list 내 파일명이 디렉터리 경로일 경우, 해당 디렉터리 내 파일들로 리스트를 재구성
        if not os.path.isfile(file_name_list[0]):
            fp_list = []
            for f in file_name_list:
                for now_f in [os.path.join(f, file) for file in os.listdir(f) if file.endswith(extension)]:
                    fp_list.append(now_f)
            file_name_list = fp_list
        # 파일 리스트가 비어있을 시, 경로가 잘못된 것으로 간주하고 에러 발생
        if len(file_name_list) == 0:
            print("경로 내, 파싱할 파일이 존재하지 않습니다.")
            raise FileExistsError()

        # column 구성
        result_list = ["Sentence #,Word,Sentiment,Aspect" + "\n"]
        now_index = 1
        # 각 file을 파싱
        for now_p in file_name_list:
            print('-' * 50)
            print("now parsing File: {name}".format(name=now_p))
            with open(now_p, encoding="UTF-8") as f:
                now_file = json.loads(f.read())

            total_example += len(now_file)

            for s_index, row in enumerate(now_file):
                words = row['RawText']

                if check_bad_example(words, s_index, 'all'):
                    example_nan_error += 1
                    continue  # example 전체 스킵

                # 문장을 띄어쓰기 기준 split
                words = sentence2words(words)
                sentiments = ['O'] * len(words)
                aspects = ['O'] * len(words)

                for t_index, span_info in enumerate(row["Aspects"]):
                    subwords_span = span_info["SentimentText"]
                    if check_bad_example(subwords_span, s_index, (t_index + 1)):
                        continue  # there is no sentiment span or there is nan value #오류만 스킵

                    subwords_span = is_sublist(sentence2words(subwords_span), words)
                    if subwords_span is None:  # there is no sub match => 매칭 되지 않는 경우
                        if remove_spacing(span_info["SentimentText"]) in remove_spacing(row['RawText']):
                            sentiment_text = remove_spacing(span_info["SentimentText"])
                            start_index = remove_spacing(row['RawText']).find(sentiment_text)

                            if start_index == -1:
                                print("{s_index} index, {t_index} tag there is no match".format(s_index=s_index + 2,
                                                                                                t_index=t_index))
                                label_match_error += 1
                                continue
                            span = [start_index, start_index + len(sentiment_text) - 1]

                            # RawText와 SentimentText에 단순히 띄어 쓰기 차이만 있을 경우
                            # re-calculation of subwords_span
                            word_index_dict = {}
                            char_idx = 0
                            for idx, w in enumerate(words):
                                for c_idx in range(len(w)):
                                    word_index_dict[char_idx] = idx
                                    char_idx += 1
                            subwords_span = (word_index_dict[span[0]], word_index_dict[span[1]] + 1)
                        else: # RawText와 SentimentText가 스펠링이 서로 다를 경우
                            print("{s_index} index, {t_index} tag there is no match".format(s_index=s_index + 2,
                                                                                            t_index=t_index))
                            label_match_error += 1
                            continue
                    sentiment = polarity_dict[span_info["SentimentPolarity"]]
                    aspect = span_info["Aspect"]
                    # 각 속성의 분포 계산
                    aspect_dist_dict = cal_distribution(aspect_dist_dict, aspect)
                    polarity_dist_dict = cal_distribution(polarity_dist_dict, sentiment)

                    if sentiment == '0' or aspect == '0':  # label 이 잘못 표시 된 경우
                        print("{s_index} index, {t_index} tag there there is wrong "
                              "sentiment label".format(s_index=s_index + 2, t_index=t_index))
                        label_error += 1
                        continue
                    # BIO tag 부착
                    for i in range(subwords_span[0], subwords_span[1]):
                        if i == subwords_span[0]:
                            sentiments[i] = 'B-' + sentiment
                            aspects[i] = 'B-' + aspect
                        else:
                            sentiments[i] = 'I-' + sentiment
                            aspects[i] = 'I-' + aspect

                now_index += s_index
                for i in range(len(words)):
                    if i == 0:
                        result_list.append(
                            "Sentence {now_index},".format(now_index=now_index) + words[i] + "," + sentiments[i] + "," +
                            aspects[i] + "\n")
                    else:
                        result_list.append("," + words[i] + "," + sentiments[i] + "," + aspects[i] + "\n")

        # 파일 생성
        with open(result_fp, 'w', encoding=args.encoding) as outfile :
            for result in result_list:
                outfile.write(result)
        total_processed_example = total_example - example_nan_error
        printing_error_size(result_fp, total_example, total_processed_example, example_nan_error,
                            label_match_error, label_error, polarity_dist_dict, aspect_dist_dict,
                            args.save_p, args.encoding)
        entire_total_example += total_example
        entire_total_processed_example += total_processed_example
        entire_example_nan_error += example_nan_error  # 입력 문장이 None 으로 인식 되는 경우
        entire_label_match_error += label_match_error  # 입력 문장과 레이블 스팬이 일치 하지 않는 경우
        entire_label_error += label_error
        entire_aspect_dist_dict += aspect_dist_dict
        entire_polarity_dist_dict += polarity_dist_dict

    print("\n" + '*' * 50)
    printing_error_size("Entire Files", entire_total_example, entire_total_processed_example, entire_example_nan_error,
                        entire_label_match_error, entire_label_error, entire_polarity_dist_dict,
                        entire_aspect_dist_dict, args.save_p, args.encoding)

