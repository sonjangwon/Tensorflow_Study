import re
import pandas as pd
import numpy as np
import gensim
import gensim.models
import sys
from gensim.models import Word2Vec
# ------------------------------
# -- 파일 읽기
# ------------------------------
file = open("fire30.txt", "r")  
sentences = file.read() 

# 문장별로 Split 처리
model_data = re.split("[\n\.?]", sentences.lower())

# -- 공백/빈 리스트 제거
while ' ' in model_data:
    model_data.remove(' ')
    model_data.remove('')
 

# -- 데이터프레임에 저장
model_dataframe = pd.DataFrame()
model_dataframe['sentences'] = np.asarray(model_data)
 

# -- 데이터프레임 문장별 Split
model_dataframe["separates"] = model_dataframe["sentences"].apply(lambda x: x.replace(" ",""))
model_dataframe["separates"] = model_dataframe["sentences"].apply(lambda x: x.replace(",",""))
model_dataframe["separates"] = model_dataframe["separates"].apply(lambda x: x.replace(";",""))
model_dataframe["separates"] = model_dataframe["separates"].apply(lambda x: x.replace("\"",""))
model_dataframe["separates"] = model_dataframe["separates"].apply(lambda x: x.replace('"',''))
model_dataframe["separates"] = model_dataframe["separates"].apply(lambda x: x.split())

# -- 문장별 Word2Vec 처리
model = Word2Vec(model_dataframe["separates"], sg=1, size=300, min_count=1, iter=10 )
'''
sg = 0이면 cbow, 1이면 skip-grm
min_count = 5  (등장 횟수가 5 이하인 단어는 무시)
size = 300 (300차원짜리 벡터스페이스에 embedding)
iter (보통 딥러닝에서 말하는 epoch와 비슷한, 반복횟수
workers : cpu의 코어수에 따라 multi-thread를 지원해서 병렬처리하는 옵션
alpha : 초기학습률, min_alpha: alpha값이 학습과정에서 선형으로 줄어서 도달하는 최소 값
''' 


count=0;
sum=0.0;
average=0.0;

## 입력받은 문장
Sentence = "mommy daddy are having fight police going baseball fuck"
print(model.doesnt_match(Sentence.split()))

for i in Sentence.split():
    print(model.similarity(i,model.doesnt_match(Sentence.split())))
    sum += model.similarity(i,model.doesnt_match(Sentence.split())) 
    count += 1
average = sum /count
print(average, sum, count) 		



word_file = open("fullvoc.txt", "r")  
words_lists = word_file.read()

def edit_distance(s1, s2):
    l1, l2 = len(s1), len(s2)
    if l2 > l1:
        return edit_distance(s2, s1)
    if l2 is 0:
        return l1
    prev_row = list(range(l2 + 1))
    current_row = [0] * (l2 + 1)
    for i, c1 in enumerate(s1):
        current_row[0] = i + 1
        for j, c2 in enumerate(s2):
            d_ins = current_row[j] + 1
            d_del = prev_row[j + 1] + 1
            d_sub = prev_row[j] + (1 if c1 != c2 else 0)
            current_row[j + 1] = min(d_ins, d_del, d_sub)
        prev_row[:] = current_row[:]
    return prev_row[-1]

#for i in words_lists.split():
    #print(edit_distance("fuck",i))
count=0;
sum=0.0;
changed_word_similarty = average
higher_word = model.doesnt_match(Sentence.split())
for i in words_lists.split():
    #print(i)
    #print(edit_distance("fuck",i)==3)
    try:
        if edit_distance(model.doesnt_match(Sentence.split()),i)==3 | edit_distance(model.doesnt_match(Sentence.split()),i)==3:
            for j in Sentence.split():
                sum += model.similarity(j,i)
                count += 1
                #print(model.doesnt_match(Sentence.split()),i,j,sum,count)
            average = sum/count
            print(i, average)
 
            if changed_word_similarty <average:
                changed_word_similarty = average
                higher_word = i
            print(higher_word)
            count =0;
            average =0.0;
            sum=0.0;
    except:
        sum=0.0;
	 
    #print(higher_word)
print(Sentence)
Sentence = Sentence.replace(model.doesnt_match(Sentence.split()),higher_word)
print(Sentence)
 


