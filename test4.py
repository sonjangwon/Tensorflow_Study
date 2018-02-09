import gensim
sentences = gensim.models.word2vec.Text8Corpus('text8')
model = gensim.models.word2vec.Word2Vec(sentences, size=200)
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
 
