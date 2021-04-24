#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:19:43 2020

@author: chasebrown
"""

from subprocess import Popen, PIPE, STDOUT
import os
import docx
import sys
import nltk
nltk.download('punkt')
import random
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import unicodedata
from datetime import datetime

#sys.stdout = open('output.txt', 'w')
now = datetime.now()
print("1. Started: " + str(now))
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

# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)

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

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

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
            if '{START}' in text:
                isNeeded = False
                for i in text.split('\n'):
                    if isNeeded:
                        if '{STOP}' in i:
                            isNeeded = False
                            line = i.replace('{STOP}', '')
                            if not line in raw_data['No']:
                                raw_data['No'].append(line)
                        else:
                            raw_data['Yes'].append(i)
                    else:
                        if '{START}' in i:
                            isNeeded = True
                            line = i.replace('{START}', '')
                            if not line in raw_data['Yes']:
                                raw_data['Yes'].append(line)
                        else:
                            raw_data['No'].append(i)
        except:
            print("Error with: " + filename)
            print(os.path.abspath('../Files/' + filename))
            print(sys.exc_info())

data = raw_data          
#data = {"Yes":raw_data['Yes'][int(len(raw_data["Yes"])/2):], "No":raw_data['No'][int(len(raw_data["No"])/2):]}
#testdata = {"Yes":raw_data['Yes'][:int(len(raw_data["Yes"])/2)], "No":raw_data['No'][:int(len(raw_data["No"])/2)]}
#raw_data = ""


# get a list of all categories to train for
categories = list(data.keys())
#for category in categories:
    #print(category)
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

#print(docs)
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
#print(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()

last = now
now = datetime.now()
print("2. Started: " + str(now) + " | Step 1 Took: " + str(now-last))


# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs', tensorboard_verbose=3)
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=25, batch_size=8, show_metric=True)

#----------------------------------------NN Trained-------------------------------------

last = now
now = datetime.now()
print("3. Started: " + str(now) + " | Step 2 Took: " + str(now-last))

directory = os.fsencode("../Files/")
numOfFiles = len(os.listdir(directory))
counter = 0
fileDataCorrect = {}
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".docx") or filename.endswith(".DOCX"):
        fileDataCorrect.update({filename:[]})
        try:
            text = " ".join(getText("../Files/" + filename))
            for i in text.split('{START}'):
                if '{STOP}' in i:
                    for s in i.split("{STOP}")[0].replace('"', '\"').split('\n'):
                        if not s.strip() == "":
                            fileDataCorrect[filename].append(s.strip())
        except:
            print("Error with: " + filename)
            print(os.path.abspath('../Files/' + filename))
            print(sys.exc_info())

last = now
now = datetime.now()
print("4. Started: " + str(now) + " | Step 3 Took: " + str(now-last))

fileDataTest = {}
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".docx") or filename.endswith(".DOCX"):
        fileDataTest.update({filename:[]})
        try:
            fileDataTest.update({filename:[]})
            newFileName = filename.split('.')[0] + ".txt"
            text = " ".join(getText("../Files/" + filename))
            file = open("./Text Docs/" + newFileName, 'w+')
            file.write(text)
            file.close()
            path = os.path.abspath("./Text Docs/" + newFileName)
            p = Popen(['java', '-jar', './LegalParser.jar', path], stdout=PIPE, stderr=STDOUT)
            found = False
            foundData = []
            for line in p.stdout:
                if "Exception in thread" in str(line)[2:-1]:
                    found = False
                if found:
                    broken = str(line)[2:-1].split(' ')
                    foundData.append({"line": int(broken[1]), "index": int(broken[4]), "ITN": int(broken[6]), "anchor": ' '.join(broken[10:])})
                else:
                    if 'Second Ordered Lists' in str(line)[2:-1]:
                        found = True
            with open(path, 'r+') as file:
                txtFileData = []
                for line in file:
                    txtFileData.append(line.replace("{START}", "").replace("{STOP}", ""))
                for point in range(0,len(foundData)-1):
                    #print(str(point))
                    text = ""
                    final = ""
                    for i in range(0,len(txtFileData)):
                        if i == foundData[point]['line']-1:
                            text += txtFileData[i][foundData[point]['index']-1:] + '\n'
                            #print("-------1------")
                            #print(str(i))
                            #print(txtFileData[i][foundData[point]['index']-1:] + '\n')
                            #print("-------1------")
                        elif i>foundData[point]['line'] and i<foundData[point+1]['line']-1:
                            if categories[np.argmax(model.predict([get_tf_record(str(txtFileData[i]))]))]=="Yes":
                                text += txtFileData[i] + '\n'
                                #print("-------2------")
                                #print(str(i))
                                #print(txtFileData[i] + '\n')
                                #print("-------2------")
                        elif i == foundData[point+1]['line']:
                            if categories[np.argmax(model.predict([get_tf_record(str(txtFileData[i][:foundData[point+1]['index']-1]))]))]=="Yes":
                                text += txtFileData[i][:foundData[point+1]['index']-1]
                                #print("-------3------")
                                #print(str(i))
                                #print(txtFileData[i][:foundData[point+1]['index']-1])
                                #print("-------3------")
                            final += txtFileData[i][foundData[point+1]['index']-1:] + '\n'
                        elif i>foundData[point+1]['line']:
                            if point == len(foundData)-1:
                                if categories[np.argmax(model.predict([get_tf_record(str(txtFileData[i]))]))]=="Yes":
                                    final += txtFileData[i] + '\n'
                    
                    fileDataTest[filename].append(text.strip())
                    if point == len(foundData)-1:
                        fileDataTest[filename].append(final.strip())
        except:
            pass
            #print("Error with: " + filename)
            #print(os.path.abspath('../Files/' + filename))
            #print(sys.exc_info())

last = now
now = datetime.now()
print("5. Started: " + str(now) + " | Step 4 Took: " + str(now-last))

fileHRData = {}
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".docx") or filename.endswith(".DOCX"):
        fileHRData.update({filename:[]})
        try:
            text = " ".join(getText("../Files/" + filename))
            for i in text.split('\n'):
                if categories[np.argmax(model.predict([get_tf_record(str(i))]))]=="Yes":
                    fileHRData[filename].append(i)
        except:
            print("Error with: " + filename)
            print(os.path.abspath('../Files/' + filename))
            print(sys.exc_info())


#----------------------------------------Files Loaded-------------------------------------

last = now
now = datetime.now()
print("6. Started: " + str(now) + " | Step 5 Took: " + str(now-last))
            
analytics_test = []
for file in fileDataCorrect.keys():
    temp = {'file': file, 'numCorrect':0, 'numExtra':0, 'numMissing':0}
    for answer in fileDataCorrect[file]:
            found = False
            for guess in fileDataTest[file]:
                if answer.strip() in guess.strip():
                    found = True
                    temp['numCorrect']+=1
                    break
            if not found:
                temp['numMissing']+=1
    for guess in fileDataTest[file]:
        found = False
        for answer in fileDataCorrect[file]:
            if guess.strip() in answer.strip():
                found = True
                break
            if not found:
                temp['numExtra']+=1
    analytics_test.append(temp)
    
last = now
now = datetime.now()
print("7. Started: " + str(now) + " | Step 6 Took: " + str(now-last)) 
    
broken = []
analytics_HR = []
for file in fileDataCorrect.keys():
    temp = {'file': file, 'numCorrect':0, 'numExtra':0, 'numMissing':0}
    for answer in fileDataCorrect[file]:
        found = False
        for guess in fileHRData[file]:
            if answer.strip() in guess.strip():
                found = True
                temp['numCorrect']+=1
                break
        if not found:
            if not file in broken:
                broken.append(fileDataCorrect[file])
            temp['numMissing']+=1
    for guess in fileHRData[file]:
        found = False
        for answer in fileDataCorrect[file]:
            if guess.strip() in answer.strip():
                found = True
                break
            if not found:
                temp['numExtra']+=1
    analytics_HR.append(temp)
        
    
    
last = now
now = datetime.now()
print("8. Started: " + str(now) + " | Step 7 Took: " + str(now-last))

correctRate = 0
extra = 0
numberEmpty = 0
counterTest = 0
for i in analytics_test:
    if not (i['numCorrect']+i['numMissing']) == 0:
        if i['numCorrect']/(i['numCorrect']+i['numMissing'])>.8:
            counterTest+=1
        correctRate += (i['numCorrect']/(i['numCorrect']+i['numMissing']))
    else:
        numberEmpty += 1
    extra += i['numExtra']
    
correctRate_HR = 0
extra_HR = 0
numberEmpty_HR = 0
counterHR = 0
for i in analytics_HR:
    if not (i['numCorrect']+i['numMissing']) == 0:
        if i['numCorrect']/(i['numCorrect']+i['numMissing'])==1:
            counterHR+=1
        correctRate_HR += (i['numCorrect']/(i['numCorrect']+i['numMissing']))
    else:
        numberEmpty_HR += 1
    extra_HR += i['numExtra']
    
print("Is 'Response: ' important information? " + categories[np.argmax(model.predict([get_tf_record("Response: ")]))])
print("|Test Model| On Average, this is " + str((correctRate/(len(analytics_test)-numberEmpty))*100) + "% correct")
print("|Test Model| On Average, there are " + str(extra/(len(analytics_test))) + " extra")
print("|Test Model| There are " + str(counterTest) + " completly correct out of " + str(len(analytics_test)))
print("|Hard Return| On Average, this is " + str((correctRate_HR/(len(analytics_HR)-numberEmpty_HR))*100) + "% correct")
print("|Hard Return| On Average, there are " + str(extra_HR/(len(analytics_HR))) + " extra")
print("|Hard Return| There are " + str(counterHR) + " completly correct out of " + str(len(analytics_HR)))

last = now
now = datetime.now()
print("Step 8 Took: " + str(now-last))


