#!/usr/bin/env python
# coding: utf-8

# ### Data import

# In[28]:


import pandas as pd
import sys

data = pd.read_json(sys.argv[1])


# ### Cleaning text

# In[ ]:


import spacy
import re
import unicodedata

def setup_abbr():
    file = open("abbr_portuguese.txt", encoding='utf-8')
    abbr_dict = {}

    for line in file:
        w = line.split(";")
        abbr_dict[w[0]] = w[1].replace("\n", "")
    file.close()

    return abbr_dict

def clean(data):
    doc = nlp(data)
    doc_lower = doc.text.lower()
    doc_without_emoji = emoji_pattern.sub(r'', doc_lower)
    doc_punctuation = u"".join([c for c in unicodedata.normalize('NFKD', doc_without_emoji) if not unicodedata.combining(c)])
    doc_corrected = nlp(" ".join([abbr_dict.get(w, w) for w in doc_punctuation.split()]))
    
    return doc_corrected.text

nlp = spacy.load('pt_core_news_sm')
abbr_dict = setup_abbr()
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F" # emoticons
                           u"\U0001F300-\U0001F5FF" # symbols & pictographs
                           u"\U0001F680-\U0001F6FF" # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

data['cleaned_reviews'] = data['text'].apply(clean)


# ### Tagging

# In[ ]:


import joblib
from nltk import word_tokenize

def wordTag(text):
    tagger = joblib.load('POS_tagger_brill.pkl')
    text = tagger.tag(word_tokenize(text))
    return text
 
data['tag_reviews'] = data['cleaned_reviews'].apply(wordTag)


# ### Tokenization

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize

def tokenize(text):
    text = word_tokenize(text)
    return text

data['tokenized_reviews'] = data['cleaned_reviews'].apply(tokenize)


# ### Stemming

# In[ ]:


import nltk
from nltk.stem import RSLPStemmer

def stemming(text):
    stemmer = RSLPStemmer()
    phrase = []
    for word in text:
        phrase.append(stemmer.stem(word))
    return phrase

data['stem_reviews'] = data['tokenized_reviews'].apply(stemming)


# ### Stopwords remove

# In[ ]:


import nltk
from nltk.corpus import stopwords

def stopwordsRemove(text):
    stop_words = stopwords.words('portuguese')
    phrase = []
    for word in text:
        if word not in stop_words:
            phrase.append(word)
    return phrase

data['stopwords_reviews'] = data['stem_reviews'].apply(stopwordsRemove)


# ### Lemmatizer

# In[ ]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    lemma = " "
    for word in text:
        lemma = wordnet_lemmatizer.lemmatize(word)
        
    return lemma

data['lemma_reviews'] = data['stopwords_reviews'].apply(lemmatize)


# ### Phrase junction

# In[ ]:


def juction(text):
    phrase = []
    for word in text:
        phrase.append(word)
    
    phraseStr = ' '.join(phrase)
    return phraseStr

data['junction'] = data['stopwords_reviews'].apply(juction)


# ### Information extraction

# In[ ]:


import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('pt_core_news_sm')

def informationExtraction(text):
    evaluations = []
    doc = nlp(text)
    
    securityTerms = ['segur', 'roub', 'clon', 'senh', 'acess']
    patterns = [nlp(term) for term in securityTerms]
    
    #Ver utilização do add_patterns juntamente com o entity_ruler (otimização)
    matcher = PhraseMatcher(nlp.vocab) 
    matcher.add("SECURITY_PATTERN", patterns)
    
    matches = matcher(doc)
    
    for i in range(0,len(matches)):
        token = doc[matches[i][1]:matches[i][2]]
        evaluations.append(str(token))
            
    return evaluations

data['extracted_reviews'] = data['junction'].apply(informationExtraction)


# In[ ]:

print("---------- OUTPUT -----------")

for i in range(len(data)):
    if len(data.loc[i,'extracted_reviews'])!=0:
        print(data.loc[i,'cleaned_reviews'])
