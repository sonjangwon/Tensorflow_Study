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
    print(word)
    # Find appropriate word index
    for i, j in enumerate(word_list):
        if j == word:
            word_index = i
    print(word_index)
    # Calculate cosine similarity with all other words
    similar_values = []
    similar_words = []
    print(vectors)
    for i, j in enumerate(vectors):
        similarity = (cosine_similarity(vectors[word_index].reshape(1, -1), vectors[i].reshape(1, -1)))[0][0]
        similar_values.append(abs(similarity))
        similar_words.append(word_list[i])
   
    # Create a dataframe with all similarities and word indices, sort it and get the most similar words
    similarities = pd.DataFrame()
    similarities['word'] = similar_words
    similarities['similarity'] = similar_values
    lower_values = 0.0
    similarities = similarities.sort_values(by=similarities.columns[1], ascending=False)
    count =0;
   # print(min(similar_values))
    #print(similar_values)
    lower_values = similar_values[0]
    #print(lower_values)
    lower_words = 'word'
    for i in enumerate(similar_values):
        if(i[1] > lower_values):
             lower_values=i[1]
             lower_words=similar_words[count]
            # print(lower_values)
             #print(lower_words)
        count+=1
        #if(lower_values > i):
        #     lower_values = i
        #     print("wow")
        #print(i[1])
        #count += 1
       # if(lower_values == i[1]):
       #      print("same")
       #      lower_words = similar_words.index(count)
       #      print(similar_words.index(count))
        #print(similar_words.index(count))
	    #if min(similar_values) == similar_values[i]:
		  #   lower_words = similar_words[i]
	
        #print(i)
#	    print(similar_values[i].type())
#	    if similar_values[i].doublevalue() < lower_values:
 #            lower_values = similar_values[i]
  #           lower_wods =  similar_words[i]
    #print(lower_values)
    #print(lower_words)
    #print(similarities.head(n=10))

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
