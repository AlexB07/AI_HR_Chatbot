import pickle
import json
import tflearn
import tensorflow
import numpy as np
import random
import numpy as np
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tkinter
from tkinter import *

#loading previous files from training the model
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
trainX = data['trainX']
trainY = data['trainY']
dataFile = r'C:\Users\alex-\Desktop\Python\ChatBotProject\intents_full.json'
with open(dataFile) as jsonData:
    intents = json.load(jsonData)

#Creating a blank model to load the previous trained model
net = tflearn.input_data(shape=[None, len(trainX[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(trainY[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('./model.tflearn')


#FUNCTIONS
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

#Bag of words procedure for user input
def bow(sentence, words, showDetails=False):
    sentenceWords = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentenceWords:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if showDetails:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

context = {'318':'employeenumber'}

ERROR_THRESHOLD = 0.25
#Works out probability of each documents over threshold value. 
def classify(sentence):  
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

#Finds the response in which the chatbot will say
def getResponse(sentence, userID='318', showDetails=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if showDetails: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if showDetails: print ('tag:', i['tag'])
                        result = random.choice(i['responses'])
                        return result

            results.pop(0)
			
#Creating the UI
#Functions in which the UI needs.


def sends():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    
    if msg !='':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        
        res = getResponse(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        
def send(event):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    
    if msg !='':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        
        res = getResponse(msg)
        if res == None:
            res = 'Sorry, I cant understand you. Please rephrase your question.'
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        
base = Tk()
base.title("HR Chatbot")
base.geometry("500x600")
base.resizable(width=FALSE, height=FALSE)
base.bind('<Return>', send)
    
#creating the chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)
  
#Binding the scrollbar
sb = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = sb.set
  
#Creating the send button
sendbtn = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                 bd=0, bg="#27e64d", activebackground='#33b04c', fg='#ffffff', command=sends)
#Creating entry box
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
    
#Connect all components together
sb.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=270)
sendbtn.place(x=276, y=401, height=90, width=120)

base.mainloop()