! pip install tweet-preprocessor

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
# nltk.download('all-corpora')  # use in case you are unsure what to download so download all

import string
import preprocessor as p
import numpy as np
import pandas as pd
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import bigrams,trigrams
from nltk.tokenize import word_tokenize,sent_tokenize
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer

tok = WordPunctTokenizer()

stopword = set(stopwords.words('english'))
stopword.update(['amp','cc','rt','@handle']) # rt = retweet

stemmer = SnowballStemmer('english')
lemma = WordNetLemmatizer()

def remove_stopwords(text):
    return ' '.join([word.lower() for word in text.split(' ') if word.lower() not in stopword and len(word)>2])

def get_vocab(data):
    vocab = {}
    for tweet in tqdm(data):
        for word in tweet.split(' '):
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def text_cleaner(text):
    
    # text = re.sub(r'^https?:\/\/.*[\r\n]*', '-URL-', text, flags=re.MULTILINE) # sub URL
    text = re.sub(r'http\S+', ' ', text) 
    text = re.sub(r'www\S+', ' ', text) 
    
    text = re.sub(r'\d',' ',text) # remove any number
    
    # text = re.sub('@handle',' ',text)
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'[^a-zA-Z]', ' ', text) 
    
    # text = re.sub(r'[:]',' ',text) # special characters
    # text = re.sub(r'[-]',' ',text)
    # text = re.sub(r'[!]',' ',text)
    # text = re.sub(r'[_]',' ',text)
    # text = re.sub('[?]',' ',text)
    # text = re.sub('[.]',' ',text)
    # text = re.sub('[+]',' ',text)
    # text = re.sub('[=]',' ',text)
    
    # text = re.sub(r'([\W_])\1+',' ',text) # multiple repeating special characters
    # text = re.sub(r'(?<=\s)[\W\d](?=(\s|$))', ' ', text)
    # text = re.sub(r'(?<=\w)\W+(?=(\s|$))', ' ', text)
    # text = re.sub(r'(\W)\1+(?=\w)', r'\1', text)
    
    text = re.sub('\n',' ',text)
    text = re.sub('\s\s+',' ',text) # multiple space
    text = text.lstrip()
    text = text.rstrip()
    return text   


contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 
"could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", 
"hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 
"how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", 
"I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 
"i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", 
"it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", 
"let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
"mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
"needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", 
"oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 
"she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
 "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
 "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
 "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
 "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
 "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
 "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
 "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  
 "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
 "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 
 "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 
 "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
 "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",
 "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
 "you'll've": "you will have", "you're": "you are", "you've": "you have", "ya'll": "you all" ,"let's":'let us'}

def remove_contraction(text):
    for word in text.split(' '):
        if word.lower() in contraction_dict:
            text = text.replace(word, contraction_dict[word.lower()])  
    return text



# https://sproutsocial.com/insights/social-media-acronyms/  # link to abbrevations

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())



def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 


def clean_tweets_within_df(df,col_name):
    
    df['tweetos'] = '' 
    
    #add tweetos first part
    for i in range(len(df[col_name])):
        try:
            df['tweetos'][i] = df[col_name].str.split(' ')[i][0]
        except AttributeError:    
            df['tweetos'][i] = 'other'

    #Preprocessing tweetos. select tweetos contains 'RT @'
    for i in range(len(df[col_name])):
        if df['tweetos'].str.contains('@')[i]  == False:
            df['tweetos'][i] = 'other'

    # remove URLs, RTs, and twitter handles
    for i in range(len(tweets['text'])):
        df[col_name][i] = " ".join([word for word in df[col_name][i].split()
                                    if 'http' not in word and '@' not in word and '<' not in word])


    df[col_name] = df[col_name].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))
    df[col_name] = df[col_name].apply(lambda x: re.sub('  ', ' ', x))



def clean(text):
    text = text.lower()
    
    # keep alphanumeric characters only
    text = re.sub('\W+', ' ', text).strip()
    text = text.replace('user', '')
    
    # tokenize
    text_token = word_tokenize(text)
    
    # replace shortcuts using dict
    full_words = []
    for token in text_token:
        if token in shortcuts.keys():
            token = shortcuts[token]
        full_words.append(token)
        
#     text = " ".join(full_words)
#     text_token = word_tokenize(text)
    # stopwords removal
#     words = [word for word in full_words if word not in stop]

    words_alpha = [re.sub(r'\d+', '', word) for word in full_words]
    words_big = [word for word in words_alpha if len(word)>2]
    stemmed_words = [lemma.lemmatize(word) for word in words_big]
    
    # join list elements to string
    clean_text = " ".join(stemmed_words)
    clean_text = clean_text.replace('   ', ' ')
    clean_text = clean_text.replace('  ', ' ')
    return clean_text




pat1 = r'@[A-Za-z0-9]+' # firsr regex pattern
pat2 = r'https?://[A-Za-z0-9./]+' # second regex pattern
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # tokenize and join together to remove unneccessary white spaces
    
    words = tok.tokenize(lower_case)
    
    return (" ".join(words)).strip()
    





emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])


emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])



emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)

# clean_text = p.clean(twitter_text)


def clean_tweets(tweet):
 
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
# after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    
#replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    
#remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    
#filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    
#looping through conditions
    for w in word_tokens:
        
#check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)

    #print(word_tokens)
    #print(filtered_sentence)return tweet
