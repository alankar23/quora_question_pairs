# %%
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit
from nltk.corpus import stopwords                   #Stopwords corpus
import re                                           # Regular expressions library
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import pattern.en as en
from sklearn.feature_extraction.text import CountVectorizer
# %%
data = pd.read_csv('advance_features.csv')
data.head()
# null_data = data[data.isnull().any(1)].index.tolist()
# data = data.drop(null_data)

data.isnull().sum()
# %%
data['question1'] = data['question1'].fillna('')
data['question2'] = data['question2'].fillna('')
data.isnull().sum()
# %%
duplicate = data[data['is_duplicate'] == 1]
nonduplicate = data[data['is_duplicate'] == 0]

# Converting 2d array of q1 and q2 and flatten the array: like {{1,2},{3,4}} to {1,2,3,4}
          
a = np.dstack([duplicate["question1"], duplicate["question2"]]).flatten()
b = np.dstack([nonduplicate["question1"], nonduplicate["question2"]]).flatten()
# %%
stopwords = set(STOPWORDS)
stopwords.add('whi')
stopwords.add('doe')
# %%
def cloud(data,count):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = 'english', min_df =5) 
    vectors = vectorizer.fit_transform(data)
    counts = vectors.sum(axis=0).A1
    vocab = list(vectorizer.get_feature_names())
    frequency = Counter(dict(zip(vocab, counts)))
    frequency_2 = frequency.most_common(count)
    words = ' '.join(x[0] for x in frequency_2)
    wordcloud = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 10,stopwords=stopwords).generate(words) 
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show() 
# %%
cloud(a,200)
cloud(b,200)

# %%
n = data.shape[0]
sns.pairplot(data[['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio', 'is_duplicate']][0:n],hue='is_duplicate',vars= ['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'])
plt.show()
# %%
