path = '../wordoccoutputs/twitter/ncaa/'
fout=open(path+"completeOutput.csv","a")
for num in range(7):
    for line in open(path+"part-0000"+str(num)+".csv"):
         fout.write(line)
fout.close()
