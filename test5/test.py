import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
 
# ------------------------------
# -- 파일 읽기
# ------------------------------
file = open("move2.txt", "r")
moby_dick = file.read()
#print(moby_dick)
 
#print("<raw_doc", "_"*100)
 
# ------------------------------
# -- 문장별로 Split 처리
# ------------------------------
moby_dick = re.split("[\n\.?]", moby_dick)
#print(moby_dick)
 
#print("<split_doc", "_" * 100)
 
# ------------------------------
# -- 공백/빈 리스트 제거
# ------------------------------
while ' ' in moby_dick:
    moby_dick.remove(' ')
    moby_dick.remove('')
 
    #print(moby_dick)
 
#print("<remove_blank_doc", "_" * 100)
 
# ------------------------------
# -- 데이터프레임에 저장
# ------------------------------
df_Mobydic = pd.DataFrame()
df_Mobydic['sentences'] = np.asarray(moby_dick)
 
#print (df_Mobydic)
 
#print("<df_doc", "_" * 100)
 
# ------------------------------
# -- 데이터프레임 문장별 Split
# ------------------------------
df_Mobydic["separates"] = df_Mobydic["sentences"].apply(lambda x: x.replace(" ",""))
df_Mobydic["separates"] = df_Mobydic["sentences"].apply(lambda x: x.replace(",",""))
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.replace(";",""))
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.replace("\"",""))
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.split())
#print(df_Mobydic["separates"])
#print (df_Mobydic)

#print("<df_sep_doc", "_" * 100)
 
# ------------------------------
# -- 문장별 Word2Vec 처리
# ------------------------------
#print(df_Mobydic["separates"])
model = Word2Vec(df_Mobydic["separates"], sg=0, size=200, min_count=1,iter=100)
#'min_count': 5,  # 등장 횟수가 5 이하인 단어는 무시
# 'size': 300,  # 300차원짜리 벡터스페이스에 embedding
#  'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용한다
#  'batch_words': 10000,  # 사전을 구축할때 한번에 읽을 단어 수
#  'iter': 10,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수
# worker: 병렬처리, 높아질수록 속도는 빨라지지만 정확도는 낮아진다.

 
#print(model.doesnt_match("we snow and high winds pounded the US east coast along a front stretching from Maine as far south as North Carolina early on Thursday, knocking out power, icing over roadways and closing hundreds of schools.".split()))
testSentence = "Hello, this is Wasington DC Station There is a fire at the starbucks building. Please come quickly."
#print(model.doesnt_match(testSentence.split()))
count =0;
#for i in testSentence.split():
    #count +=1
    #if model.doesnt_match(testSentence.split()) == i:
        #print(i, testSentence.split()[count])
        #print(model.similarity(i, testSentence.split()[count]))
try:	
    print(model.similarity("Hello", "this"))
    print(model.similarity("this", "is"))
    print(model.similarity("is", "Wasington"))
    print(model.similarity("Wasington", "DC"))
    print(model.similarity("DC", "Station"))
    print(model.similarity("Station", "There"))
    print(model.similarity("There", "is"))
    print(model.similarity("is", "a"))
    print(model.similarity("a", "fire"))
except:
    print("except")
finally:
    print("finally")
#print(model.most_similar("gun"))
#for word, score in model.most_similar("Washington"):
#    print(word, score)
#print(model.similarity('protein','Starbucks'))
#print(model.similarity('plaze','his'))
#print(model.similarity('shot','brazil'))
