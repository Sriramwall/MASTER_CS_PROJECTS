import pandas as pd
import json
sources=['cc', 'nyc', 'twitter']
options=['nba','ncaa']

all_top_words={ 'ccnba': ['nba','game','team','part','sport','turner','play','player','point','last'],
           'ccncaa': ['sport','illustr','favorit','sign','pleas','servic','custom','team','point','list'],
           'nycnba': ['point','game','first','score','play','coach','team','second','go','lead'],
           'nycncaa': ['point','first','game','state','play','second','tournament','no.','coach','team'],
           'twitternba': ['nba','playoff','game','player','get','nbaplayoff','win','team','watch','like'],
           'twitterncaa': ['ncaa','tournament','game','final','duke','basketbal','marchmad','elit','team','win']}



for source in sources:
    for option in options:
        filename = source+'/'+option+'/completeOutput.csv'
        df = pd.read_csv(filename)
        top_words = all_top_words[source+option]
        for w in top_words:
            df = df[df['word'] != '<'+w+',the>']
            df = df[df['word'] != '<'+w+',I>']
            df = df[df['word'] != '<'+w+',|>']
            df = df[df['word'] != '<'+w+',get>']
            df = df[df['word'] != '<'+w+',but>']
            df = df[df['word'] != '<'+w+',-->']
            df = df[df['word'] != '<'+w+',—>']
            df = df[df['word'] != '<'+w+',->']
            df = df[df['word'] != '<'+w+',_>']
            df = df[df['word'] != '<'+w+',___>']
            df = df[df['word'] != '<'+w+',- >']
            found=df[df['word'].str.startswith('<'+w+',')]
            found = found.sort_values(by=["count"], ascending=False)
            res = found.head(20).values
            for i in range(len(res)):
                res[i][0]=res[i][0].replace('<'+w+',', '').replace('>', '')
            dat = pd.DataFrame(res, columns = ['word', 'count'])
            with open('jsonFiles/'+source+'_'+option+'_'+w+'.json', 'w') as f:
                json.dump({"children": dat.to_dict(orient='records')}, f, indent=4)
