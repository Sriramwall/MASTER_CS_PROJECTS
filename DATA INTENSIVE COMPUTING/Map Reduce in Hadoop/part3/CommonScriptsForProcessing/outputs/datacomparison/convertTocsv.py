import csv, os
path = 'nyc_full/'
Filenames = next(os.walk(path))[2]
print(Filenames)
for file in Filenames:
    txt_file = file
    csv_file = file+'.csv'
    in_txt = csv.reader(open(path+txt_file, "rt", encoding='utf-8'), delimiter = '\t')
    out_csv = csv.writer(open(path+csv_file, 'wt', encoding='utf-8'))
    out_csv.writerows(in_txt)
