import csv
import os

data_dirpath = '/home/satoshi/dev/kaggle/spooky/data'
csv_filepath = os.path.join(data_dirpath, 'test.csv')
#csv_filepath = os.path.join(data_dirpath, 'train.csv')
text_dirpath = os.path.join(data_dirpath, 'test')

with open(csv_filepath, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        if len(row) == 3:
            out_filename = '{}_{}.txt'.format(row[0], row[2])
        else:
            out_filename = '{}_.txt'.format(row[0])
        with open(os.path.join(text_dirpath, out_filename), 'w') as out_f:
            out_f.write(row[1])

