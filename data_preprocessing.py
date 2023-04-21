import re
def Remove_Speaker(corpus):
    corpus=corpus.replace("--", " ", -1)
    corpus=corpus.split("\n")
    pre_corpus = []
    for line in corpus:
        # Searching for Orator names using re
        x = re.search("^([A-Za-z])+\s*([A-Za-z])*:$", line)
        if (x==None):
            pre_corpus.append(line)
    print("Remove_Speakers - Done")
    return pre_corpus

def TotalWords(corpus,tokenizer):
    total_words = []
    for line in corpus:
        tokens = tokenizer(line)
        total_words.extend(tokens)
    print("Total - Done")
    return total_words

def UniqueWords(total_words):
    unique_words={}
    for word in total_words:
        if word in unique_words:
            unique_words[word]+=1
        else:
            unique_words[word]=1
    print("Unique Words - Done")
    return unique_words

def IntegerTokening(unique_words):
    ids_vocabulary = {}
    ids = 0
    for word, v in unique_words.items():
        ids_vocabulary[word] = ids
        ids += 1
    print("Int Token - Done")
    return ids_vocabulary

def Tokenize(corpus, ids_vocab):
    tokenized_corpus = []
    for line in corpus:
        new_line = []
        for word in line:
            if word in ids_vocab:
                new_line.append(ids_vocab[word])
        if len(new_line) > 1:
            tokenized_corpus.append(new_line)
    print("Tokenizing Done")
    return tokenized_corpus


def SplitIndices(n, val_pct):
    n_val = int(val_pct * n)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

def GetVocabulary(data,tokenizer):
    num_samples = 0
    dataset = []
    vocabulary
    for label, review in data:
        chars_to_remove = ['--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ']
        for chars in chars_to_remove:
            review = review.replace(chars, " ", -1)
        num_samples += 1
        if label == "neg":
            label = 0
        else:
            label = 1
        tokens = tokenizer(review)
        vocabulary.extend(tokens)
        dataset.append((tokens, label))
    return dataset, num_samples,vocabulary







