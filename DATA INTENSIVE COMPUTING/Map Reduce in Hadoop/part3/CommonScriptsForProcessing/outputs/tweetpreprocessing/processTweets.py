import pandas as pd
from nltk.stem import PorterStemmer as ps
from nltk.corpus import stopwords
stemmer = ps()

stop_words = set(stopwords.words('english'))

def stemAndRemoveStop(paras):
    tempRes=[]
    paras=str(paras)
    if paras:
        for w in paras.split(' '):
            if w not in stop_words:
                tempRes.append(stemmer.stem(w))
        return ' '.join(tempRes)


ncaa = pd.read_csv("ncaa_tweets.csv")
ncaaTweets = ncaa.text
ncaaTweets = ncaaTweets.apply(stemAndRemoveStop)
ncaaTweets.to_csv('ncaa_tweets.txt', index=False, sep=' ', header=None)


nba = pd.read_csv("nba_tweets2.csv")
nbaTweets = nba.text
nbaTweets = nbaTweets.apply(stemAndRemoveStop)
nbaTweets.to_csv('nba_tweets2.txt', index=False, sep=' ', header=None)
