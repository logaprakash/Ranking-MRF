# -*- coding: utf-8 -*-
"""
Created on Tue May  8 06:34:07 2018

@author: logaprakash
"""

from collections import defaultdict
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import LDA
from pyspark.sql import SQLContext
import re
import numpy as np
import csv
from time import time

a = time()

def document_vector(document):
    id = document[1]
    counts = defaultdict(int)
    for token in document[0]:
        if token in vocabulary:
            token_id = vocabulary[token]
            counts[token_id] += 1
    counts = sorted(counts.items())
    keys = [x[0] for x in counts]
    values = [x[1] for x in counts]
    return (id, Vectors.sparse(len(vocabulary), keys, values))


sc = SparkContext('local', 'PySPARK LDA')
sql_context = SQLContext(sc)

data = sc.wholeTextFiles('/home/sakshi/Documents/big_data/data/*').map(lambda x: x[1])

num_of_stop_words = 50      
num_topics = 15	            
num_words_per_topic = 100    
max_iterations = 35         


tokens = data                                                   \
    .map( lambda document: document.strip().lower())            \
    .map( lambda document: re.split("[\s;,#]", document))       \
    .map( lambda word: [x for x in word if x.isalpha()])        \
    .map( lambda word: [x for x in word if len(x) > 3] )


termCounts = tokens                             \
    .flatMap(lambda document: document)         \
    .map(lambda word: (word, 1))                \
    .reduceByKey( lambda x,y: x + y)            \
    .map(lambda tuple: (tuple[1], tuple[0]))    \
    .sortByKey(False)
    
threshold_value = termCounts.take(num_of_stop_words)[num_of_stop_words - 1][0]

vocabulary = termCounts                         \
    .filter(lambda x : x[0] < threshold_value)  \
    .map(lambda x: x[1])                        \
    .zipWithIndex()                             \
    .collectAsMap()

documents = tokens.zipWithIndex().map(document_vector).map(list)
#inv_voc = {value: key for (key, value) in vocabulary.items()}

lda_model = LDA.train(documents, k=num_topics, maxIterations=max_iterations)
topic_indices = lda_model.describeTopics(maxTermsPerTopic=num_words_per_topic)
topic_document_matrix = lda_model.topicsMatrix()
document_data_df = documents.map(lambda x: (x[0], x[1])).toDF(("DocID","Word_Counts"))
word_data = document_data_df.select('Word_Counts').rdd.map(lambda x: x[0])   

"""
Vectorizing
"""
document_data_list = word_data.collect()
num_docs = len(document_data_list)
vocab_len = len(vocabulary)

vectors = np.zeros((num_docs,num_topics))

for document in range(0,num_docs):
    for topic in range(0,num_topics):
        for word in range(0,vocab_len):
            if document_data_list[document][word] != 0.0:
                vectors[document][topic] += document_data_list[document][word] * topic_document_matrix[word][topic]



"""
Saving required frames
"""
np.savetxt("/home/sakshi/Documents/big_data/vectors.csv", vectors, delimiter=",")
np.savetxt("/home/sakshi/Documents/big_data/topic_word_matrix.csv", topic_document_matrix, delimiter=",")

with open('/home/sakshi/Documents/big_data/vocabulary.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, vocabulary.keys())
    w.writeheader()
    w.writerow(vocabulary)

"""
Ending session
"""
sc.stop()

b = time() -a