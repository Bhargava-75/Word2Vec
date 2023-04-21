import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
from Models import Encoder,Decoder,Skip_Gram
from data_preprocessing import Remove_Speaker,Tokenize,TotalWords,UniqueWords,IntegerTokening
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
device = torch.device('cuda:1')

print("Started")
corpus = open("tiny_shakes.txt", "r").read()
corpus = Remove_Speaker(corpus)
tokenizer = get_tokenizer("basic_english")
total_words = TotalWords(corpus,tokenizer)
unique_words= UniqueWords(total_words)
vocabulary_ids = IntegerTokening(unique_words)

num_words = len(unique_words)
model = Skip_Gram(num_words,300)
opt = torch.optim.Adam(model.parameters(),lr=1e-4)
model.load_state_dict(torch.load('skip_gram.pt'))
model.eval()



shakes_vectors = {}
total_words=[]
for x, v in vocabulary_ids.items():
    word = torch.zeros(num_words)
    word[v] = 1
    word=word.to(device)
    total_words.append(x)
    shakes_vectors[x] = model.get_embeddings(word).cpu().detach().numpy()
print("Storing Done")



print("Classification Starts")
# Load the data
data = pd.read_csv('movie_reviews.csv')
X=data['review']
Y=data['sentiment']

# Preprocess the text data
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print("Preprocessing start")
X = X.apply(preprocess)
print("Preprocessing done")



def vectorize(sentence):
    words = sentence.split()
    words_vecs = [shakes_vectors[word] for word in words if word in total_words]
    if len(words_vecs) == 0:
        return np.zeros(100)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)

#Finding Vectors
print("Started Data vectorizing")
X = np.array([vectorize(sentence) for sentence in tqdm(X)])
print("Ended Data vectorizing")


print("Started Training")
results = open("Word2Vec results.txt","w")
size = 0.8
accuracy=[]
precision=[]
recall=[]
f1_scores=[]
for i in range(4):
    print(f"Step {i+1}")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=42)

    # Train a classification model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, pos_label='positive')
    rec = recall_score(y_test, y_pred, pos_label='positive')
    f1  = f1_score(y_test, y_pred, pos_label='positive')
    accuracy.append(round(acc,4))
    precision.append(round(pre,4))
    recall.append(round(rec,4))
    f1_scores.append(round(f1,4))
    size=size-0.1

print(accuracy)
print(precision)
print(recall)
print(f1_scores)

results.write("\n\nAccuracy : "+str(accuracy)+"\n")
results.write("Precision : "+str(precision)+"\n")
results.write("Recall : "+str(recall)+"\n")
results.write("F1-Score : "+str(f1_scores)+"\n")

results.close()
