# skip_ - statistical skipgram vectorizer
# version: 0.1
# author: Michal Pikusa (pikusa.michal@gmail.com)

import time
import sys
import nltk
from nltk import skipgrams
import pandas as pd
import scipy
import collections
import numpy as np
import pickle

def main():
    # Start the counter
    start = time.time()

    # Load raw data and tokenize
    print("Loading data...")
    corpus_file = str(sys.argv[1])
    corpus = open(corpus_file, 'r')
    text = corpus.readlines()
    text = list(map(str.strip, text))
    text_string = ' '.join(text)
    print("Tokenizing...")
    tokens = nltk.word_tokenize(text_string)

    # Function to create a dataframe with counts and probabilities
    def create_count_df(list_to_count):
        list_with_counts = collections.Counter(list_to_count)
        df = pd.DataFrame()
        df['word'] = list_with_counts.keys()
        df['count'] = list_with_counts.values()
        df['prob'] = df['count'] / sum(df['count'])
        return df

    # Create the list of unigrams with the count and normalize probability
    print("Creating the list of unigrams...")
    unigram_df = create_count_df(tokens)
    print("Creating the list of skipgrams...")
    skipgram_list = list(skipgrams(tokens, 2, 2))
    skipgram_df = create_count_df(skipgram_list)
    print("# tokens: ", len(tokens))
    print("# unigrams: ", unigram_df.shape[0])
    print("# skipgrams: ", skipgram_df.shape[0])

    # For each pair of words calculate the PMI and create a data frame
    print("Calculating PMI values for each skipgram...")
    skipgram_df[['word1', 'word2']] = skipgram_df['word'].apply(pd.Series)
    unigram_df = unigram_df.set_index('word')
    skipgram_df['prob1'] = skipgram_df['word1'].map(unigram_df['prob'].get)
    skipgram_df['prob2'] = skipgram_df['word2'].map(unigram_df['prob'].get)
    skipgram_df['pmi'] = np.log(skipgram_df['prob'] / (skipgram_df['prob1'] * skipgram_df['prob2']))
    skipgram_df = skipgram_df[['word1', 'word2', 'pmi']]

    # Pivot the data frame into a sparse matrix, and convert NaNs into 0s
    print("Converting into a matrix...")
    pmi_matrix = skipgram_df.pivot(index='word1', columns='word2', values='pmi')
    pmi_matrix = pmi_matrix.fillna(0)

    # Apply SVD to reduce the size of the matrix to get word vectors
    print("Extracting word vectors...")
    U, S, V = scipy.sparse.linalg.svds(pmi_matrix, k=int(sys.argv[2]))
    word_list = unigram_df.index.get_values()

    # Save the model
    print("Saving model...")
    word_list_name = '_'.join([sys.argv[3], 'wordlist.p'])
    vectors_name = '_'.join([sys.argv[3], 'vectors.p'])
    output_word_list = open(word_list_name, 'wb')
    pickle.dump(word_list, output_word_list)
    output_word_list.close()
    output_vectors = open(vectors_name, 'wb')
    pickle.dump(U, output_vectors)
    output_vectors.close()

    # Print out overall statistics of the run
    end = time.time()
    print("Running time: ", str(round(end - start, 1)), "seconds")
    return

if len(sys.argv) < 4:
    print("Missing arguments!")
    print("Correct syntax: python create_vectors.py <text_file> <number_of_vectors> <model_name>")
    print("Example: ")
    print("python create_vectors.py academic.txt 300 academic")
else:
    main()