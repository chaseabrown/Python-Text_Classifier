import docx
import os
import sys

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

directory = os.fsencode("../Files/")
numOfFiles = len(os.listdir(directory))
counter = 0
output = open("data.csv", "a+")
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
                if '{STOP}' in i:
                    output.write("\"" + i.split("{STOP}")[0].replace('"', '\"') + "\",\"" + filename + "\"\n")
        except:
            print("Error with: " + filename)
            print(os.path.abspath('../Files/' + filename))
            print(sys.exc_info())
output.close()


directory = os.fsencode("../Files/")
numOfFiles = len(os.listdir(directory))
counter = 0
fileData = {}
for file in os.listdir(directory)[-2:]:
    filename = os.fsdecode(file)
    if filename.endswith(".docx") or filename.endswith(".DOCX"):
        fileData.update({filename:[]})
        try:
            words = getText("../Files/" + filename)
            text = " ".join(words)
            for i in text.split('{START}'):
                if '{STOP}' in i:
                    fileData[filename].append(i.split("{STOP}")[0].replace('"', '\"'))
            
        except:
            print("Error with: " + filename)
            print(os.path.abspath('../Files/' + filename))
            print(sys.exc_info())
            
            
            
            
            
            