import csv, os
Filenames = next(os.walk('.'))[2]
print(Filenames)
for file in Filenames:
    txt_file = file
    csv_file = file+'.csv'
    in_txt = csv.reader(open(txt_file, "rt", encoding='utf-8'), delimiter = '\t')
    out_csv = csv.writer(open(csv_file, 'wt', encoding='utf-8'))
    out_csv.writerows(in_txt)
