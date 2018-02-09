import os
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities

os.chdlr("C:\Users\Jangwon\Documents\Tensorflow\04 - Neural Network Basic");
df=pd.read_csv("train2.csv");

x=df['question1'].values.tolist()
y=df['question2'].values.tolist()

corpus = x+y

tok_corp = [nltk.word_tokenize(sent.decode('utf-8')) for sent in corpus]

model = gensim.models.Word2Vec(tok_corp, min_count=1,size=32)
