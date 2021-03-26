# %%
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pycontractions import Contractions
from fuzzywuzzy import fuzz
from tqdm import tqdm
import distance
import numpy as np
from wordcloud import WordCloud
# %%
data = pd.read_csv('simple_features.csv')
data.head()

# %%
def punctutions(data):
    data = re.sub(r'[^\w\s]', '', data)
    data = re.sub(r"([0-9]+)000000", r"\1m", data)
    data = re.sub(r"([0-9]+)000", r"\1k", data)
    return data


# %%
# %%
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def stemming(sentence):
    stemmer = PorterStemmer()
    sentence = sentence.split()
    sentence = ' '.join(stemmer.stem(word) for word in sentence ) #if word not in stop_words)
    return sentence

# %%
cont = Contractions(api_key="glove-twitter-100")
# %%

data['question1'] = list(cont.expand_texts(data['question1']))
data['question2'] = list(cont.expand_texts(data['question2']))
data['question1'] = data['question1'].fillna('').apply(lambda x: BeautifulSoup(x, "lxml").text)
data['question2'] = data['question2'].fillna('').apply(lambda x: BeautifulSoup(x, "lxml").text)
data['question1'] = data['question1'].fillna('').apply(punctutions)
data['question2'] = data['question2'].fillna('').apply(punctutions) 
data['question1'] = data['question1'].fillna('').apply(stemming)
data['question2'] = data['question2'].fillna('').apply(stemming)

#%%
data['fuzz_ratio'] = data.apply(lambda x : fuzz.ratio(x['question1'],x['question2']),axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(x['question1'], x['question2']),axis=1)
data['token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(x['question1'],x['question2']),axis=1)

# # %%

# %%
cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max, last_word_eq, first_word_eq, abs_len_diff, mean_len, longest_substr_ratio = [],[],[],[],[],[],[],[],[],[],[]

#%%
Safe_add = 0.0001
for x in tqdm(range(int(data.shape[0]))):
    token_1 = data.question1[x].split()
    token_2 = data.question2[x].split()

        # Get non stop words:
    words_1 = set([word for word in token_1 if word not in stop_words])
    words_2 = set([word for word in token_2 if word not in stop_words])

        # Get common stop words:
    stop_q1 = set([word for word in token_1 if word in stop_words])
    stop_q2 = set([word for word in token_2 if word in stop_words])

        # Get the common non-stopwords from Question pair
    common_word_count = len(words_1.intersection(words_2))
        
        # Get the common stopwords from Question pair
    common_stop_count = len(stop_q1.intersection(stop_q2))
        # Get the common Tokens from Question pair
    common_token_count = len(set(token_1).intersection(set(token_2)))

    cwc_min.append(common_word_count / (min(len(words_1), len(words_2))+Safe_add))
    cwc_max.append(common_word_count / (max(len(words_1), len(words_2))+Safe_add))
    csc_min.append(common_stop_count / (min(len(stop_q1), len(stop_q2))+Safe_add))
    csc_max.append(common_stop_count / (max(len(stop_q1), len(stop_q2))+Safe_add))
    ctc_min.append(common_token_count / (min(len(token_1),len(token_2))+Safe_add))
    ctc_max.append(common_token_count / (max(len(token_1),len(token_2))+Safe_add))
    
    if len(token_1)== 0 or len(token_2) == 0:
        last_word_eq.append(0)
    elif token_1[-1] == token_2[-1]:
        last_word_eq.append(1)
    else:
        last_word_eq.append(0)
    
    if len(token_1)== 0 or len(token_2) == 0:
        first_word_eq.append(0)
    elif token_1[0] == token_2[0]:
        first_word_eq.append(1)
    else:
        first_word_eq.append(0)
    
    abs_len_diff.append(abs(len(token_1)-len(token_2)))
    mean_len.append((len(token_1)+len(token_2) / 2))

    strs = list(distance.lcsubstrings(data['question1'][x],data['question2'][x]))
    if len(strs) ==0:
        longest_substr_ratio.append(0)
    else:
        longest_substr_ratio.append(len(strs[0])/min(len(data['question1'][x]),len(data['question2'][x])))
# %%
data['cwc_min'] = cwc_min
data['cwc_max'] = cwc_max
data['csc_min'] = csc_min
data['csc_max'] = csc_max
data['ctc_min'] = ctc_min
data['ctc_max'] = ctc_max
data['last_word_eq'] = last_word_eq
data['first_word_eq'] = first_word_eq
data['abs_len_diff'] = abs_len_diff
data['mean_len'] = mean_len
data['longest_subs_ratio'] = longest_substr_ratio
# %%
# %%
data.to_csv("advance_features.csv", index=False)
# %%

# %%