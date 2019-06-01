import requests
#import nltk
#nltk.download('stopwords')
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer as ps
from nltk.corpus import stopwords
stemmer = ps()

apiKey = 'RLqjAlE1UAf803j7EDqD9iGSyCLTQVrW'
queryWord = 'NCAA'
queryWord.replace(' ', '+')
url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q='
beginDate = '20190315'
stop_words = set(stopwords.words('english'))

def getInitialURL():
    return url+queryWord+'&begin_date='+beginDate+'&api-key='+apiKey

def getURL(page):
    print(url+queryWord+'&page='+str(page)+'&sort=oldest&begin_date='+beginDate+'&api-key='+apiKey)
    return url+queryWord+'&page='+str(page)+'&sort=oldest&begin_date='+beginDate+'&api-key='+apiKey

def getData(url):
    data = requests.get(url, timeout=10)
    content = BeautifulSoup(data.content, "html.parser")
    return content

def stemAndRemoveStop(paras):
    res = []
    for para in paras:
        tempRes = []
        for w in para.split(' '):
            if w not in stop_words:
                tempRes.append(stemmer.stem(w))
        res.append(' '.join(tempRes))
    return res

response = requests.get(getInitialURL())
data = response.json()
numberOfHits = int(data['response']['meta']['hits']/10)

collection = []
for page in range(9, numberOfHits):
    print(page)
    response = requests.get(getURL(page))
    data = response.json()
    articlesUrls = list(map(lambda d: d['web_url'], data['response']['docs']))

    for u in articlesUrls:
        content = getData(u)
        paras= content.find_all("p")
        collection.append(' '.join(stemAndRemoveStop(list(map(lambda c: c.text, paras)))))

for i, c in enumerate(collection):
    collection[i]=c.replace("advertis support By the associ press ", '')
    collection[i]=c.replace("advertis support By ", '')
    collection[i]=c.replace("advertis support ", '')
    collection[i]=c.replace("advertis", '')
    file = open('nytimesNCAAArticle/Article'+str(i)+'.txt', 'w')
    file.write(collection[i])
    file.close()



#import os, sys
#k=340
#for i in range(0, 89):
#    os.rename('Article'+str(i)+'.txt', 'Article'+str(k)+'.txt')
#    k+=1

#import shutil
#for i in range(340, 429):
#    shutil.move("Article"+str(i)+'.txt', "nytimesNCAAArticle/"+"Article"+str(i)+'.txt')    
