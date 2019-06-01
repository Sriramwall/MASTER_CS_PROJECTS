path = 'twitter_partial/'
fout=open(path+"twitter_partial.csv","a")
for num in range(3):
    for line in open(path+"part-0000"+str(num)+".csv"):
         fout.write(line)
fout.close()
