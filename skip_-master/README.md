# ![alt text](https://github.com/michal-pikusa/skip_/blob/master/skip_logo.png)
Skip_ is a statistical skipgram vectorizer. It transforms words into vectors in multidimensional space, based on point mutual information. Contrary to other popular vectorizers like word2vec, it does not require a neural network to run, as it works on unigram/skipgram frequencies and creates desired multidimensional space by applying singular-value decomposition. Skip_ saves the resulting vector model into a file that can be used for subsequent operations, like e.g. finding nearest neighbors.
# Usage
There are two scripts in the repository, one for building the vector model, and the other for using the model to find similar words. An example text corpus of sample academic texts is included to get you started. 
To build your model on the included text with 300-dimensional vectors, simply type:
```
python create_vectors.py academic.txt 300 academic
```
Now, in order to find the nearest neighbors in the text to e.g. word "stress", type:
```
python find_nn.py academic stress
```
which will show you ten words with highest cosine similarity mesaure:
```
              word    similarity
161         stress    1.000000
4211  psychiatrist    0.527578
2086        animal    0.489718
1934          teen    0.417422
6811     caucasian    0.410263
1802      despacio    0.368893
6148          prod    0.368663
6449        kirson    0.358575
3715     inclusion    0.358166
6727          bond    0.337418
```

# Misc
Author: Michal Pikusa (pikusa.michal@gmail.com)

Version: 0.1
