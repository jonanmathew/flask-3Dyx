import nltk
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

nltk.download('maxent_ne_chunker')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('large_grammars')
nltk.download('snowball_data')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

with open('./chatbot/chat.json', 'r') as f:
    chat = json.load(f)

wml = WordNetLemmatizer()


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


all_words = []
tags = []
xy = []

for i in chat['chat']:
    tag = i['tag']
    tags.append(tag)
    for pattern in i['patterns']:
        w = tokenizer(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = [',', '!', '?', '.']

all_words = [lemmatize(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sent, tag) in xy:
    bag = bag_of_words(pattern_sent, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 10
dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


hidden_size = 16
output_size = len(tags)
input_size = len(all_words)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

learning_rate = 0.001
num_epochs = 800

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer
        optimizer.zero_grad()  # set gradiants to zero
        loss.backward()  # calculate back propogation
        optimizer.step()  # performs a parameter update based on the current gradient

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
