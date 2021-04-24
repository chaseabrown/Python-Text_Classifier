import sys
#sys.stdout = open('output.txt', 'w')
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
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
    

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs', tensorboard_verbose=3)
model.load('model.tflearn')


#Testing Model

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


allTests = []
directory = os.fsencode("./Files/")
numOfFiles = len(os.listdir(directory))
counter = 0
for file in os.listdir(directory)[:10]:
    testdata = {"Yes":[], "No":[]}
    counter += 1
    gettingGroup = False
    filename = os.fsdecode(file)
    if filename.endswith(".docx") or filename.endswith(".DOCX"):
        try:
            line = ""
            lines = []
            categories = []
            words = getText("./Files/" + filename)
            text = " ".join(words)
            for i in text.split('{START}'):
                if not i.isspace():
                    if '{STOP}' in i:
                        before = i.split("{STOP}")[0]
                        after = i.split("{STOP}")[1]
                        for line in before.split("\n"):
                            if not line in raw_data['Yes']:
                                testdata['Yes'].append(line)
                        for line in after.split("\n"):
                            if not line in raw_data['No']:
                                testdata['No'].append(line)
                    else:
                        for line in i.split("\n"):
                            if not line in raw_data['Yes']:
                                testdata['No'].append(line)
        except:
            print("Error with: " + filename)
            print(os.path.abspath('files/' + filename))
            print(sys.exc_info())
        allTests.append({'data': testdata, 'doc': filename})

counter = 1
for test in allTests:
    print(counter)
    counter+=1
    # get a list of all categories to test for
    testcategories = list(test['data'].keys())
    # a list of tuples with words in the sentence and category name
    testdocs = []
    
    for each_category in test['data'].keys():
        print("In Loop")
        for each_sentence in test['data'][each_category]:
            # remove any punctuation from the sentence
            each_sentence = remove_punctuation(each_sentence)
            testdocs.append((each_sentence, each_category))
    print("out of Loop")
    
    numCorrect = 0;
    numsIncorrect = []
    testCounter = 1;
    
    printDoc = False
    for doc in testdocs:
        if(doc[1] == categories[np.argmax(model.predict([get_tf_record(doc[0])]))]):
            print("Correct")
            numCorrect += 1
        else:
            printDoc = True
            print("Wrong")
            break
    if printDoc:
        print("Making File")
        htmlfile = open("(" + test['doc'] + ")errors.html", "a+")
        htmlfile.write("""
        <html>
            <body>            
                <table>
        """)
        words = getText("./Files/" + test['doc'])
        text = " ".join(words)
        for i in text.split('\n'):
            htmlfile.write("<tr><td>" + i + "</td><td>" + categories[np.argmax(model.predict([get_tf_record(remove_punctuation(i))]))] + "</td></tr>")
            
        htmlfile.write("""
                </table>
            </body> 
        </html>
        """)
        htmlfile.close()
            
            
        
#print(str(numCorrect) + " Out of " + str(testCounter) + " correct. " + str(round((numCorrect/testCounter),3)) + " percent correct.")

