fout=open("completeOutput.csv","a")
for num in range(7):
    for line in open("part-0000"+str(num)+".csv"):
         fout.write(line)
fout.close()
