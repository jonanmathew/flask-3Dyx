import random
import json
import torch
from chatbot.calc import get_num
from chatbot.converter import converter
from chatbot.translator import translator
from chatbot.spellcheck import spell
from chatbot.webs import find_source,scrape
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./chatbot/chat.json', 'r') as json_data:
    chats = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE) 
ans=""
cho=""

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.l3=nn.Linear(hidden_size,num_classes)
        self.relu=nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

wml=WordNetLemmatizer()

def tokenizer(sentence):
    return nltk.word_tokenize(sentence)

def lemmatize(words):
    return wml.lemmatize(words.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [lemmatize(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag
        
def chatbot(sentence):
        query=sentence

        sentence = tokenizer(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.85:
            for c in chats['chat']:
                if tag == c["tag"] and tag!="calculator" and tag!="spell" and tag!="converter" and tag!="translator" and tag!="classiojrfn":
                    return random.choice(c['responses'])
        else:
                ans=scrape(query)
                return(ans)
                exit()

        if tag=="calculator" or re.search(r"(\d+)\s*([+*/-])\s*(\d+)", query):
            ans=get_num(query)
            return(ans)

        if tag=="converter":
            user_input = query
            try:
                parts = user_input.split()
                value = float(parts[1])
                unit_from = parts[2].lower()
                unit_to = parts[4].lower()
                result = converter.convert(unit_from, unit_to, value)
                ans=(f"{value} {unit_from} is equal to {result} {unit_to}")
                return(ans)
            except (ValueError, IndexError):
                return("Invalid input format")


        if tag=="translator":
            words = query.split()
            new_string = " ".join(words[1:-2])
            dest = words[-1]
            ans=translator(new_string,dest)
            return ans

        if tag=="classiojrfn":
            return("I can help you with the following : \n...Your doubts regarding Classio \n...Calculator [<num> <+,-,*,/,**,%,sin,cos,tan,root> <num>] \n...Converter[<num> <metric> to <new> <metric>] \n...Translator[translate <text> to <destination language>] \n...Spell Checker[spell check <sentence>]")

        if tag=="spell":
            ans=spell(query)
            return(ans)
