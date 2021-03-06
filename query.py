#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:21:22 2018

@author: sakshi
"""

from collections import defaultdict
from pyspark.mllib.linalg import Vectors
import numpy as np
import csv
import glob
from time import time
import os
import subprocess as sb

def document_vector(document):
    id = document[1]
    counts = defaultdict(int)
    for token in document.split(' '):
        if token in vocabulary:
            token_id = vocabulary[token]
            counts[token_id] += 1
    counts = sorted(counts.items())
    keys = [x[0] for x in counts]
    values = [x[1] for x in counts]
    
    return (id, Vectors.sparse(len(vocabulary), keys, values))

def getQueryVector(query):
    query_word_count = document_vector(query.lower())
    query_list = list(query_word_count)[1]
    
    query_vector = np.zeros((num_topics))
    for topic in range(0,num_topics):
        for word in range(0,vocab_len):
            if query_list[word] != 0.0:
                query_vector[topic] += query_list[word] * topic_word_matrix[word][topic]
    
    return query_vector

def nMostSimilarDocuments(query_vector_arg,no_of_results_arg):
    dist_matrix = np.zeros((num_docs,num_topics))
    for dist_index in range(num_docs):
        dist_matrix[dist_index] = vectors[dist_index] - query_vector_arg
    
    distances = np.sum(dist_matrix**2, axis=1)
    return distances.argsort()[:no_of_results_arg]

def readFileLocations(path):
    file_paths = []
    files = glob.glob(path)
    for element in files:
        file_paths.append(element)
    return file_paths

def displayResultPaths(result_data):
    print("The recommendations are...\n")
    
    result_paths = []
    for result in result_data:
        result_paths.append(file_paths[result])
        print(file_paths[result])
    
    return result_paths
    
def openResultsGedit(result_paths):
    for path in result_paths:       
        proc = sb.Popen(['gedit', path])
        proc.wait()
        

"""
Reading model data
"""
dict_reader = csv.DictReader(open('/home/sakshi/Documents/big_data/vocabulary.csv', 'r'))
vocabulary = []
for line in dict_reader:
    vocabulary.append(line)
vocabulary = vocabulary[0]

vectors = np.genfromtxt('/home/sakshi/Documents/big_data/vectors.csv', delimiter=',')
vectors = vectors/np.amax(vectors)
topic_word_matrix = np.genfromtxt('/home/sakshi/Documents/big_data/topic_word_matrix.csv', delimiter=',')

"""
Declaring variables
"""
num_docs = vectors.shape[0]
num_topics = topic_word_matrix.shape[1]
vocab_len = len(vocabulary)
no_of_results = 3
path = '/home/sakshi/Documents/big_data/data/*/*'
file_paths = readFileLocations(path)

"""
Main program
"""
query = input("Enter the query:\n")
query_vec = getQueryVector(query)
#query_vec = query_vec/max(query_vec)
results = nMostSimilarDocuments(query_vec,no_of_results)
#print(results)
result_paths = displayResultPaths(results)
openResultsGedit(result_paths)
