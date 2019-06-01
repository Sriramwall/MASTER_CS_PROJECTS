import pandas as pd
path = 'TWITTER/NCAA/'
df = pd.read_csv(path+'completeOutput.csv')
df = df.sort_values(by=["count"], ascending=False)
print(df.head(10))
