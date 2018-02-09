import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
import gensim
import numpy as np
import pandas as pd
import csv
import collections
import smart_open
import random
from pprint import pprint  # pretty-printer
#set directory
os.chdir('C:\Users\Jangwon\Documents\Tensorflow\test5')

#function definition
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]) 


#file open_learning data file
f=open('move2.txt','r')
train_corpus = list(read_corpus(f))

#for check
#train_corpus[6]

#build a model
model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)
#time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

#save a model and word, sentence vector
model.save('doc2vecmodel_20180130.model')
model.save_word2vec_format('wordvec_20180130', doctag_vec=True, word_vec=True, prefix='*dt_', fvocab=None, binary=False)

#load the model
#model = gensim.models.Doc2Vec.load('E:\Dropbox\☆SAO연구\_codetest\doc2vecmodel_20180130.model')  # you can continue training with the loaded model!

#for test
pprint(model.most_similar(u'technique', topn=20))
pprint(model.most_similar(positive=[u'', u''], negative=[u'']))
#[출처] doc2vec_python_gensim|작성자 ddonae

