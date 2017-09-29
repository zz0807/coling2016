import csv
import os
import string
import pickle
all = []
sub_dirs = [x for x in os.listdir('test_data') if os.path.isdir('test_data/'+ x)]
for sub_dir in sub_dirs:
    excel_files = [x for x in os.listdir('test_data/' + sub_dir) if os.path.isfile('test_data/' + sub_dir + '/' + x)]
    for excel_file in excel_files:
        with open('test_data/' + sub_dir + '/' + excel_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                single = {}
                single['act_tag'] = row['act_tag'];
                single['text'] = row['text'].translate(string.maketrans("" , ""), string.punctuation)
                all.append(single)

with open('test_data.pkl', 'wb') as file:
    pickle.dump(all, file)


