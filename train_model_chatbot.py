import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
dataFile=r'C:\Users\alex-\Desktop\Python\ChatBotProject\intents_full.json'
with open(dataFile) as jsonData:
    intents = json.load(jsonData)
	
words = []
classes = []
docs = []
#Removes unwanted words or puncuation.
ignoreWords = ['?','\'']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        docs.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignoreWords]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(docs), "docs")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

training = []
output = []
outEmpty = [0] * len(classes)

for doc in docs:
    bag = []
    patternWords = doc[0]
    patternWords = [stemmer.stem(word.lower()) for word in patternWords]
    for w in words:
        bag.append(1) if w in patternWords else bag.append(0)

    outRow = list(outEmpty)
    outRow[classes.index(doc[1])] = 1

    training.append([bag, outRow])

random.shuffle(training)
training = np.array(training)

trainX = list(training[:,0])
trainY = list(training[:,1])

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(trainX[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(trainY[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(trainX, trainY, n_epoch=1100, batch_size=8, show_metric=True)
model.save('model.tflearn')

pickle.dump( {'words':words, 'classes':classes, 'trainX':trainX, 'trainY':trainY}, open( "training_data", "wb" ) )
