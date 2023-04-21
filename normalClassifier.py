import pandas as pd
from sklearn.model_selection import train_test_split
import string
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# Load the data
data = pd.read_csv('movie_reviews.csv')
X=data['review']
Y=data['sentiment']

# mask_pos = Y == 'positive'
# mask_neg = Y == 'negative'

# Y = np.where(mask_pos, 1, Y)
# Y = np.where(mask_neg, -1, Y)

results = open("Linear Classifier results without word2vec.txt","a")
size = 0.8

accuracy=[]
precision=[]
recall=[]
f1_scores=[]
for i in range(4):
    print(f"Step {i+1}")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Train a classification model
    clf = LogisticRegression()
    clf.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test_vec)
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