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
file = open("life_suicide_gun.txt", "r")  
moby_dick = file.read()
#print(moby_dick)

#print("<raw_doc", "_"*100)
 
# ------------------------------
# -- 문장별로 Split 처리
# ------------------------------
moby_dick = re.split("[\n\.?]", moby_dick.lower())
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
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.replace('"',''))
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.split())
#print(df_Mobydic["separates"])
#print (df_Mobydic)

#print("<df_sep_doc", "_" * 100)
 
# ------------------------------
# -- 문장별 Word2Vec 처리
# ------------------------------
#print(df_Mobydic["separates"])
model = Word2Vec(df_Mobydic["separates"], sg=0, size=300, min_count=1, iter=10 )
print(model)
#'min_count': 5,  # 등장 횟수가 5 이하인 단어는 무시
#window : (기본값 5) 대상 단어와 대상 단어 주변의 최대 거리.
# 'size': 300,  # 300차원짜리 벡터스페이스에 embedding
#  'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용한다
#  'batch_words': 10000,  # 사전을 구축할때 한번에 읽을 단어 수
#  'iter': 10,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수
# worker: 병렬처리, 높아질수록 속도는 빨라지지만 정확도는 낮아진다. 학습하는 동안 사용할 스레드 수
#size 는 feature 벡터의 차원입니다.
#window는 문서의 내에서의 예측을 위한 예측된 단어와 문맥의 단어들 사이의 최대 거리입니다.
#dm은 트레이닝 알고리즘으로 distributed memory가 default 값입니다.
#alpha값은 초기 학습률(learning rate)이고 min_alpha는 alpha값이 학습과정에서 선형으로 줄어들어서 도달하는 최소 값입니다.
#min_count 이하의 total frequency를 가진 단어들은 모두 무w시됩니다.
#workers는 cpu의 코어 수에 따라 multi-threads를 지원해서 병렬처리하는 옵션입니다.
#size = 3000, window = 10, dm = 0, alpha=0.025, min_alpha=0.025, min_count=5, workers=multiprocessing
'''
Architecture: 아키텍처 옵션은 skip-gram (default) 와 continuous bag of words가 있다. skip-gram이 미세하게 느리지만 더 좋은 결과를 보여준다.
Training algorithm: hierarchical softmax (default) 와 negative sampling이 있다. 여기서는, 디폴트가 좋다.
Downsampling of frequent words: 구글 도큐먼트에서 .00001에서 .001 사이의 값을 추천한다. 여기서는, 0.001에 가까운 값이 좋아 보인다.
Word vector dimensionality: 많은 특성(feature)은 더 많은 학습시간을 요구하지만, 보통 더 좋은 결과를 낸다(항상 그런것은 아니다). 수십에서 수백 정도가 적당한 값이다; 우리는 300개의 특성을 사용한다.;
Context / window size: word2vec은 어떤 단어 주변의 단어들, 즉 문맥을 고려해서 해당 단어의 의미를 파악한다. 이 때 얼마나 많은 단어를 고려해야 할까? 10 정도가 hierarchical softmax에 적당하다. 이 값도 어느정도까지는 높을수록 좋다.
Worker threads: 패러렐 쓰레드의 수. 컴퓨터마다 다르겠지만, 일반적으로 4~6 정도가 적당하다.
Minimum word count: meaningful word를 규정하는 최소 word count. 이 수치 미만으로 등장하는 단어는 무시한다. 10에서 100 사이의 값이 적당하다. 우리의 경우, 각 영화가 30번씩 등장하므로, 영화 제목에 너무 많은 의미 부여를 피하기 위해 minimum word count를 40으로 설정하였다. 그 결과로 vocabulary size는 약 15,000개의 단어다.
'''
 


#print(model.doesnt_match("we snow and high winds pounded the US east coast along a front stretching from Maine as far south as North Carolina early on Thursday, knocking out power, icing over roadways and closing hundreds of schools.".split()))
count=0;
testSentence = "wasingtondc Station fire starbucks building come"
for i in testSentence.split():
    if model.doesnt_match(testSentence.lower().split()) == i:
        if testSentence.split()[0] == i:
            print(model.similarity(testSentence.split()[0],testSentence.split()[1]))
        else :
            print(model.similarity(i, testSentence.split()[count-1]))
    count +=1
			
			
			
print(model.doesnt_match(testSentence.lower().split()))
print(model.similarity("building", "come"))
print(model.similarity("station", "wash"))
print(model.similarity("fire", "star"))
print(model.similarity("fire", "statue"))
print(model.similarity("fire", "stone"))
print(model.similarity("starbucks", "fair"))
print(model.similarity("starbucks", "five"))
print(model.similarity("building", "star"))
#print(model.similarity("building", "bugs"))
print(model.similarity("building", "shouts"))
print(model.similarity("come", "build"))
print(model.similarity("come", "about"))

#for i in testSentence.split():
    #count +=1
    #print(model.similarity(come, building))
    #print(model.similarity(come, WasingtonDC))
    #if model.doesnt_match(testSentence.split()) == i:
        #print(i, testSentence.split()[count])
        #print(model.similarity(i, testSentence.split()[count]))
 
#print(model.most_similar("gun"))
#for word, score in model.most_similar("Washington"):
#    print(word, score)
#print(model.similarity('protein','Starbucks'))
#print(model.similarity('plaze','his'))
#print(model.similarity('shot','brazil'))
