#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:13:12 2020

@author: chasebrown
"""

import sys
#sys.stdout = open('output.txt', 'w')
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import unicodedata
import docx
import os

def getText(filename):
    fullText = []
    doc = docx.Document(filename)
    for para in doc.paragraphs:
        words = para.text.split() 
        for word in words:
            fullText.append(word)
        if(para.text != ""):
            fullText.append('\n')
    return fullText

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)

# initialize the stemmer
stemmer = LancasterStemmer()
# variable to hold the Json data read from the file
data = None

# read the json file and load the training data

raw_data = {"Yes":[], "No":[]}
directory = os.fsencode("../Files/")
numOfFiles = len(os.listdir(directory))
counter = 0
for file in os.listdir(directory):
    counter += 1
    gettingGroup = False
    filename = os.fsdecode(file)
    if filename.endswith(".docx") or filename.endswith(".DOCX"):
        try:
            line = ""
            lines = []
            categories = []
            words = getText("../Files/" + filename)
            text = " ".join(words)
            for i in text.split('{START}'):
                if not i.isspace():
                    if '{STOP}' in i:
                        before = i.split("{STOP}")[0]
                        after = i.split("{STOP}")[1]
                        for line in before.split("\n"):
                            if not line in raw_data['Yes']:
                                raw_data['Yes'].append(line)
                        for line in after.split("\n"):
                            if not line in raw_data['No']:
                                raw_data['No'].append(line)
                    else:
                        for line in i.split("\n"):
                            if not line in raw_data['Yes']:
                                raw_data['No'].append(line)
        except:
            print("Error with: " + filename)
            print(os.path.abspath('../Files/' + filename))
            print(sys.exc_info())
            
data = {"Yes":raw_data['Yes'][int(len(raw_data["Yes"])/2):], "No":raw_data['No'][int(len(raw_data["No"])/2):]}
testdata = {"Yes":raw_data['Yes'][:int(len(raw_data["Yes"])/2)], "No":raw_data['No'][:int(len(raw_data["No"])/2)]}
raw_data = ""


# get a list of all categories to train for
categories = list(data.keys())
for category in categories:
    print(category)
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)
        docs.append((w, each_category))


# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))


# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)

print(docs)
for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)
print(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs', tensorboard_verbose=3)
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=10, batch_size=8, show_metric=True)

#Testing Model

# a method that takes in a sentence and list of all words
# and returns the data in a form the can be fed to tensorflow
def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))

#Importing Test set

#testdata = testdata

# get a list of all categories to test for
testcategories = list(testdata.keys())
# a list of tuples with words in the sentence and category name
testdocs = []

for each_category in testdata.keys():
    for each_sentence in testdata[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        testdocs.append((each_sentence, each_category))

numCorrect = 0;
numsIncorrect = []
testCounter = 1;


for doc in testdocs:
    if(doc[1] == categories[np.argmax(model.predict([get_tf_record(doc[0])]))]):
        numCorrect += 1
    else:
        print("Number: " + str(testCounter))
        print("Text: " + doc[0])
        print("Correct: " + doc[1])
        print("Guess: " + categories[np.argmax(model.predict([get_tf_record(doc[0])]))])
        print()
    testCounter+= 1

print(str(numCorrect) + " Out of " + str(len(testdocs)) + " correct. " + str(round((numCorrect/len(testdocs)),3)) + " percent correct.")
model.save('model.tflearn')
