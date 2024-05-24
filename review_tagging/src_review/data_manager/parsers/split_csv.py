import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
import time

def print_dist(d, flag):
    print(d.head())
    asp = pd.total_dfFrame([dict(Counter(d["Aspect"]))]).T
    pol = pd.total_dfFrame([dict(Counter(d["Sentiment"]))]).T
    print("\n" + '*' * 50)
    print("About {example}\t".format(example=flag))
    print("<감정 레이블 분포>")
    print(pol)
    print("<속성 레이블 분포>")
    print(asp)


def file_split(args):
    # 파싱할 파일 list 구성
    file_list = [os.path.join(args.fp, file) for file in os.listdir(args.fp) if "csv" in file]
    # 파일을 저장할 경로가 없다면 생성
    if not os.path.exists(args.save_p):
        os.makedirs(args.save_p)

    # 저장 경로 내, 학습/검증/테스트 디렉터리 생성
    save_dir_name = ["train", "valid", "test"]
    for p in save_dir_name:
        if not os.path.exists(os.path.join(args.save_p, p)):
            os.makedirs(os.path.join(args.save_p, p))
    # print(file_list)    
    start = 0
    total_df = pd.read_csv(file_list[start])
    while total_df.empty:
        start += 1
        total_df = pd.read_csv(file_list[start])
    print(total_df)
    for file in file_list[start+1:]:
        last_num=total_df['Sentence #'].iloc[-1]        
        last_num=int(last_num.replace("Sentence ",""))
        df=pd.read_csv(file)
        if df.empty:
            continue
        
        Sentence_list=df['Sentence #'].unique()
        for index, sentence in enumerate(Sentence_list, start=last_num+1):
            df.loc[df['Sentence #'] == sentence, 'Sentence #'] = "Sentence "+str(index)
        total_df=pd.concat([total_df, df])

    """0인 애들 언더 샘플링"""
    # total_df.to_csv(args.fp+"_total.csv")
    Aspect_list = total_df['Aspect'].unique().tolist()
    Aspect_list.remove('O')
    max=0
    for Aspect in Aspect_list:
        num=len(total_df[total_df['Aspect']==Aspect])
        if(num>max):
            max=num
    
    total_df['Sentence_num'] = total_df['Sentence #'].str.extract(r'(\d+)', expand=False).astype(int)
    grouped = total_df.groupby('Sentence_num')['Aspect'].apply(list)
    filtered_grouped = grouped[grouped.apply(lambda x: all(item == 'O' for item in x))]
    # 문장 전체가 o인 애들의 sentence 번호
    filtered_grouped.index
    filtered_data = total_df[total_df['Sentence_num'].isin(filtered_grouped.index)]
    grouped = filtered_data.groupby('Sentence_num')['Aspect'].apply(list)
    grouped.sample(max).index.sort_values()
    o_data= total_df[total_df['Sentence_num'].isin(grouped.sample(max).index)]
    not_o_data = total_df[~total_df['Sentence_num'].isin(filtered_grouped.index.sort_values())]
    total_df=pd.concat([o_data, not_o_data])
    grouped = total_df.groupby('Sentence_num')['Aspect'].apply(list)
    total_df= total_df[total_df['Sentence_num'].isin(grouped.index.sort_values())]
    
    # Sentnece ID를 기준으로 group화하여 test set 랜덤 추출
    test_split = GroupShuffleSplit(test_size=args.test_ratio, n_splits=1,
                                       random_state=42).split(total_df, groups=total_df['Sentence #'])
    train_val_idxs, test_idxs = next(test_split)

    train_val = total_df.iloc[train_val_idxs]
    test = total_df.iloc[test_idxs]

        # Sentnece ID를 기준으로 group화하여 validation set 랜덤 추출
    val_split = GroupShuffleSplit(test_size=args.val_ratio,
                                      n_splits=1, random_state=42).split(train_val, groups=train_val['Sentence #'])
    train_idxs, val_idxs = next(val_split)
    train = train_val.iloc[train_idxs]
    val = train_val.iloc[val_idxs]

    
    
        # 결과 파일 저장
    train.to_csv(os.path.join(args.save_p, save_dir_name[0], save_dir_name[0] + ".csv"), encoding=args.encoding, index=False)
    val.to_csv(os.path.join(args.save_p, save_dir_name[1], save_dir_name[1] + ".csv"), encoding=args.encoding, index=False)
    test.to_csv(os.path.join(args.save_p, save_dir_name[2], save_dir_name[2] + ".csv"), encoding=args.encoding, index=False)















