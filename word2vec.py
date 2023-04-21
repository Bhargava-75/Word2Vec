import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchtext.data import get_tokenizer
import matplotlib.pyplot as plt
import random, os
import numpy as np
import re
from torch.utils.data import DataLoader
from Models import Encoder,Decoder,Skip_Gram
from sklearn.manifold import TSNE
from data_preprocessing import Remove_Speaker,Tokenize,TotalWords,UniqueWords,IntegerTokening
device = torch.device('cuda:1')

#Generating SkipGram Inputs
def GenerateSkipGramInput(token_corpus,num_words,window_size):
    skip_gram_input=[]
    for ids, line in enumerate(token_corpus):
        for word in range(len(line)):
            target = torch.zeros(num_words)
            context = torch.zeros(num_words)
            # One hot encoding of target word
            target[line[word]] = 1
            # Multi hot encoding of context words in the window
            start_index = max(0, word - window_size)
            end_index = min(len(line), word + window_size + 1)
            for i in range(start_index,end_index):
                if(i==word):
                    continue
                context[line[i]] = 1
            skip_gram_input.append((target, context))
    print("Skip Gram Input - Done")
    return skip_gram_input


#Loss Function

def loss_fn(pred, yb):
    yb = yb.to(torch.double)
    pred = pred.to(torch.double)
    loss_batch = - torch.mul(yb, torch.log(pred)).sum(dim = 1)
    loss = torch.mean(loss_batch)
    return loss

#Setting PyTorch Things
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ['PYTHONHASHSEED'] = str(seed)

#Cleaning and getting the Data
print("Started")
corpus = open("tiny_shakes.txt", "r").read()
corpus = Remove_Speaker(corpus)
tokenizer = get_tokenizer("basic_english")
total_words = TotalWords(corpus,tokenizer)
unique_words= UniqueWords(total_words)
vocabulary_ids = IntegerTokening(unique_words)
token_corpus = Tokenize(total_words, vocabulary_ids)

print("Preprocessing Done")
#Making Input for Skip Gram
window_size = 2 
num_words = len(unique_words)
skip_gram_input = GenerateSkipGramInput(token_corpus,num_words,window_size)

# Collates each skip-gram input samples into batches
def collate_fn(token_corpus):
    num_words = len(unique_words)
    batch_skip_input = torch.zeros(len(token_corpus), num_words).to(device)
    batch_skip_context = torch.zeros(len(token_corpus), num_words).to(device)
    for idx, xy in enumerate(token_corpus):
        x, y = xy
        batch_skip_input[idx, (x==1).nonzero()] = 1
        batch_skip_context[idx, (y==1).nonzero()] = 1
    return (batch_skip_input, batch_skip_context)

#Spliting data into batches
batch_size = 200
train_loader = DataLoader(skip_gram_input, batch_size, collate_fn=collate_fn)

print("Started Training\n\n")
model = Skip_Gram(num_words,300)
opt = torch.optim.Adam(model.parameters(),lr=1e-4)

total_epoch =20
model.to(device)
model.train()

loss_train = []
for epoch in range(1, total_epoch+1):
    epoch_loss = 0
    print("Epoch: ", epoch)
    iterator = tqdm(train_loader)
    for xb, yb in iterator:
        pred = model.forward(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        epoch_loss += loss.item()
        iterator.set_postfix(loss = loss.item())
    print("Loss: ", epoch_loss / len(train_loader))  
    loss_train.append(epoch_loss / len(train_loader)) 

#Saving the Model
torch.save(model.state_dict(), 'skip_gram.pt')

#Plotting Training Loss vs Epoch
epoch_x = [x for x in range(1, total_epoch+1)]
plt.plot(epoch_x, loss_train, color = 'r')
plt.title("Training cost vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.savefig("Loss vs Epoch.png")


print("Training Done")
train_results = []

model.to('cpu')
words = []
shakes_vectors = {}
vectors = open("Vectors.txt","w")
for x, v in vocabulary_ids.items():
    word = torch.zeros(num_words)
    word[v] = 1
    train_results.append( model.get_embeddings(word).detach().numpy())
    words.append(x)
    shakes_vectors[x] = model.get_embeddings(word)
vectors.write(str(shakes_vectors))
vectors.close()

print("Storing Done")


# T-SNE visualization on word embeddings
tsne = TSNE(n_components=2, perplexity=40)
train_x = tsne.fit_transform(train_results)
plt.figure(figsize=(15, 10), facecolor="azure")
num_words_to_visualize = 0
# Threshold to limit the number of words to display in plot
threshold = 300
for x, v in vocabulary_ids.items():
    plt.scatter(train_x[v, 0], train_x[v, 1])
    plt.text(train_x[v, 0], train_x[v, 1], x)
    num_words_to_visualize += 1
    if num_words_to_visualize > threshold:
        break
plt.title("T-SNE word embeddings")
plt.show()


