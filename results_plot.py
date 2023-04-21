
import matplotlib.pyplot as plt
import numpy as np

fig , ax = plt.subplots(2,2,figsize=(15,15))

#Results of Linear Classifier with Word2Vec embeddings
Accuracy1 = [0.6944, 0.6815, 0.7081, 0.7074]
Precision1 = [0.7574, 0.6257, 0.7467, 0.7447]
Recall1 = [0.5722, 0.902, 0.6304, 0.6324]
F1_Score1 = [0.6519, 0.7389, 0.6836, 0.684]

#Results of Linear Classifier with count vectors
Accuracy2=[0.8695, 0.8752, 0.8812, 0.8852]
Precision2=[0.8629, 0.8682, 0.8739, 0.8777]
Recall2 = [0.8787, 0.8844, 0.8911, 0.8954]
F1_Score2 = [ 0.8707, 0.8762, 0.8824, 0.8865]

x=20
K=[]
for i in range(4):
    K.append("Size = "+str(x)+"%")
    x=x+10
    
barWidth = 0.25
br1 = np.arange(len(K))
br2 = [x + barWidth for x in br1]


ax[0][0].set_title("Accuracy", fontweight ='bold', fontsize = 20)
ax[0][0].set_xlabel("Training Size", fontweight ='bold', fontsize = 15)
ax[0][0].set_ylabel("Accuracy", fontweight ='bold', fontsize = 15)
ax[0][0].bar(br1,Accuracy1, color ='#f58f71', width = barWidth,
            edgecolor ='grey', label ='Word2Vec')
ax[0][0].bar(br2,Accuracy2, color ='lightgreen', width = barWidth,
            edgecolor ='grey', label ='CountVec')
ax[0][0].legend()
ax[0][0].set_xticks([r+0.125 for r in range(len(K))], K,fontsize ='10')

ax[0][1].set_title("Precision", fontweight ='bold', fontsize = 20)
ax[0][1].set_ylabel("Precision", fontweight ='bold', fontsize = 15)
ax[0][1].set_xlabel("Training Size", fontweight ='bold', fontsize = 15)
ax[0][1].bar(br1,Precision1, color ='#f58f71', width = barWidth,
            edgecolor ='grey', label ='Word2Vec')
ax[0][1].bar(br2,Precision2, color ='lightgreen', width = barWidth,
            edgecolor ='grey', label ='CountVec')
ax[0][1].legend()
ax[0][1].set_xticks([r+0.125 for r in range(len(K))], K,fontsize ='10')


ax[1][0].set_title("Recall", fontweight ='bold', fontsize = 20)
ax[1][0].set_ylabel("Recall", fontweight ='bold', fontsize = 15)
ax[1][0].set_xlabel("Training Size", fontweight ='bold', fontsize = 15)
ax[1][0].bar(br1,Recall1, color ='#f58f71', width = barWidth,
            edgecolor ='grey', label ='Word2Vec')
ax[1][0].bar(br2,Recall2, color ='lightgreen', width = barWidth,
            edgecolor ='grey', label ='CountVec')
ax[1][0].legend()
ax[1][0].set_xticks([r+0.125 for r in range(len(K))], K,fontsize ='10')

ax[1][1].set_title("F1-Score", fontweight ='bold', fontsize = 20)
ax[1][1].set_ylabel("F1-Score", fontweight ='bold', fontsize = 15)
ax[1][1].set_xlabel("Training Size", fontweight ='bold', fontsize = 15)
ax[1][1].bar(br1,F1_Score1, color ='#f58f71', width = barWidth,
            edgecolor ='grey', label ='Word2Vec')
ax[1][1].bar(br2,F1_Score2, color ='lightgreen', width = barWidth,
            edgecolor ='grey', label ='CountVec')
ax[1][1].legend()
ax[1][1].set_xticks([r+0.125 for r in range(len(K))], K,fontsize ='10')
plt.savefig("Metrics.png")
