# skip_ - statistical skipgram vectorizer
# version: 0.1
# author: Michal Pikusa (pikusa.michal@gmail.com)

import time
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def main():
    # Start the counter
    start = time.time()

    # Load data
    print("Loading data...")
    word_list_name = '_'.join([sys.argv[1], 'wordlist.p'])
    vectors_name = '_'.join([sys.argv[1], 'vectors.p'])
    with open(word_list_name, 'rb') as wl:
        word_list = pickle.load(wl)
    with open(vectors_name, 'rb') as vs:
        vectors = pickle.load(vs)
    word = sys.argv[2]

    # Find appropriate word index
    for i, j in enumerate(word_list):
        if j == word:
            word_index = i

    # Calculate cosine similarity with all other words
    similar_values = []
    similar_words = []
    for i, j in enumerate(vectors):
        similarity = (cosine_similarity(vectors[word_index].reshape(1, -1), vectors[i].reshape(1, -1)))[0][0]
        similar_values.append(abs(similarity))
        similar_words.append(word_list[i])

    # Create a dataframe with all similarities and word indices, sort it and get the most similar words
    similarities = pd.DataFrame()
    similarities['word'] = similar_words
    similarities['similarity'] = similar_values
    similarities = similarities.sort_values(by=similarities.columns[1], ascending=False)
    print(similarities.head(n=10))

    # Print out overall statistics of the run
    end = time.time()
    print("Running time:", str(round(end - start, 1)),"seconds")
    return

if len(sys.argv) < 3:
    print("Missing arguments!")
    print("Correct syntax: python find_nn.py <model_name> <word>")
    print("Example: ")
    print("python find_nn.py academic stress")
else:
    main()
