
# coding: utf-8

# In[1]:


import glob

import numpy as np
import pandas as pd


# In[2]:


df = pd.DataFrame()

for npy in sorted(glob.iglob('data/features/*.npy')):
    df1 = pd.DataFrame(np.load(npy)).T
    df1['name'] = npy[-21:-7]
    df = pd.concat([df, df1], ignore_index=True)
    
df['class'] = df.index/80
df['class'] = df['class'].astype('int')
df


# # Why stratified train-val-test split?
# 
# We split the dataset classwise instead of randomly into 50-25-25 splits, because of the chance that a class could be underrepresented in the randomized split. If there are unequal representations of classes between train and test sets, the model trained on the train set would not perform as well on the test set. This is especially pronounced on a relatively smaller dataset such as this one, where the number of samples from each class is fairly small and equal. An extreme example could be one class not being represented at all in the training set for the randomized split, resulting in no training being done for sample of that class, resulting in poor classification for that class.

# In[3]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, train_size=0.5, test_size=0.5, stratify=df['class'])
val, test = train_test_split(test, train_size=0.5, test_size=0.5, stratify=test['class'])

with open('splits/train.npy', 'wb') as np_file:
    np.save(np_file, train)
with open('splits/val.npy', 'wb') as np_file:
    np.save(np_file, val)
with open('splits/test.npy', 'wb') as np_file:
    np.save(np_file, test)

val.sort_values('class')


# In[4]:


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

c_list = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]
scores = []

for c in c_list:
    svm = OneVsRestClassifier(estimator=SVC(C=c, kernel="linear"))
    svm.fit(train.iloc[:, :-2].values, train.iloc[:, -1].values)
    scores.append(svm.score(val.iloc[:, :-2].values, val.iloc[:, -1].values))
    
best_c = c_list[np.argmax(scores)]
best_score = np.max(scores)
print("Best c={}, score={}".format(best_c, best_score))


# In[5]:


from sklearn.metrics import classification_report, accuracy_score

best_svm = OneVsRestClassifier(estimator=SVC(C=best_c, kernel="linear"))
best_svm.fit(pd.concat([train.iloc[:, :-2], val.iloc[:, :-2]]).values, pd.concat([train.iloc[:, -1], val.iloc[:, -1]]).values)

test_pred = best_svm.predict(test.iloc[:, :-2])
print("final test accuracy: {}".format(accuracy_score(test.iloc[:, -1], test_pred)))
print(classification_report(test.iloc[:, -1], test_pred))


# In[7]:


import math
import matplotlib.pyplot as plt

failed = np.where(test.iloc[:, -1] != test_pred)[0]

fig = plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=1)

num_rows = math.ceil(len(failed)/4)

results = []
for i in range(len(failed)):
    img = test.iloc[failed[i]]['name']
    pred = test_pred[failed[i]]
    actual = test.iloc[failed[i]]['class']
    results.append({'image':  img,
                    'pred':   pred,
                    'actual': actual})

results = sorted(results, key=lambda k: k['image'])

for i in range(len(results)):
    fig.add_subplot(num_rows, 4, i+1)
    plt.xticks([]), plt.yticks([])
    image =  '{}'.format(results[i]['image'])
    pred =   'Predicted: {}'.format(results[i]['pred'])
    actual = 'Actual: {}'.format(results[i]['actual'])
    plt.xlabel(image + '\n' + pred + '\n' + actual)
    plt.imshow(plt.imread('data/images/' + results[i]['image']))
plt.show()

