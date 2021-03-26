# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
# %%
data = pd.read_csv('train.csv')
# %%
print(f'Number of data points in train set {data.shape[0]}')
# %%
# %%
data.info()

# %%
duplicate = data.groupby('is_duplicate')['id'].count()
plt.figure(figsize=(5,5))
plt.pie(duplicate, labels= ['Non-Duplicate','Duplicate'], autopct='%1.1f%%')
plt.title('Duplicate vs Non-Duplicate values')
plt.show()
# %%
questions = data['qid1'].tolist() + data['qid2'].tolist()
values = np.unique(questions)
# %%
store = {}
values = []
for x in questions:
    if x in store:
        try:
            store[x] += 1 
        except:
            pass
    else:
        store[x] = 1
# %%
max_n = store[max(store, key= store.get)]
# %%
counts = 0
for x in store.values():
    if x <= 1:
        pass
    else:
        counts += 1
print(f'number of unique questions {len(store)}')
print(f'maximum time a question appears more then once {max_n}')
print(f'Total number question appear more then once {counts}')
# %%
# store is a dict contains questions and their frequcey
# count contains number of questions appear more then once

plt.bar(['unique', 'repeated'], [len(store),counts], color=['r','b'])
plt.xlabel('Types of questions')
plt.ylabel('Frequency')
plt.show
# %%
duplicates = data[data.duplicated(['qid1','qid2'])]
print(f'Number of duplicate rows {duplicates.shape[0]}')
# %%
# %%
plt.figure(figsize= (20,10))
plt.hist(store.values(), bins = 170)
plt.yscale('log')
plt.xlabel('Frequency of questions')
plt.ylabel('Number of questions')
plt.show

# %%
data.isnull().sum()
# %%
null_data = data[data.isnull().any(1)].index.tolist()
# %%
null_data
# %%
data = data.drop(null_data)
# %%
data.isnull().sum()
# %%
"""
freq_qid1 = Frequency of qid1's
freq_qid2 = Frequency of qid2's
q1len = Length of q1
q2len = Length of q2
q1_n_words = Number of words in Question 1
q2_n_words = Number of words in Question 2
word_Common = (Number of common unique words in Question 1 and Question 2)
word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2
word_share__ = (word_common)/(word_Total)
"""
# %%
def matcher(a,b):
    a,b = a.lower().strip().split(' '), b.lower().strip().split(' ')
    return len(list(set(a).intersection(b)))

def simple_feat(data):
    data['freq_q1'] = [ store[count] for count in data['qid1']]
    data['freq_q2'] = [ store[count] for count in data['qid2']]
    data['len_q1'] = [len(sent) for sent in data['question1']]
    data['len_q2'] = [len(sent) for sent in data['question2']]
    data['words_q1'] = [len(sent.split(' ')) for sent in data['question1']]
    data['words_q2'] = [len(sent.split(' ')) for sent in data['question2']]
    data['common_words'] = [matcher(sent1,sent2) for sent1, sent2 in zip(data['question1'],data['question2'])]    
    data['total_words'] = [words1 + words2 for words1, words2 in zip(data['words_q1'],data['words_q2'])]
    #data['word_share'] = [1.0 * round(common/len(sent1.lower().strip().split(' '))+len(sent2.lower().strip().split(' ')),2) for sent1, sent2, common in zip(data['question1'],data['question2'],data['common_words'])]
    data["word_share"] = [common_w / total_w for common_w, total_w in zip(data['common_words'], data['total_words'])]
    data['freq_sum'] = [sent1+sent2 for sent1, sent2 in zip(data['freq_q1'],data['freq_q2'])]
    data['freq_dif'] = [sent1-sent2 for sent1, sent2 in zip(data['freq_q1'], data['freq_q2'])]
    data.to_csv("simple_features.csv", index=False)
    return data


# %%
simple_feat(data)
data.info()
# %%
print('Question with minimum length in question1', min(data['len_q1']))
print('Question with minimum length in question2', min(data['len_q2']))
print('Question with maximum length in question1', max(data['len_q1']))
print('Question with maximum length in question2', max(data['len_q2']))
# %%

