# Word2Vec 모델을 간단하게 구현해봅니다.
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gensim

# 단어 벡터를 분석해볼 임의의 문장들
with open('test.txt', 'r') as infile:
    sentences = np.array(infile.readlines())
