import pickle
import os
from collections import Counter

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_pickle(file_path):
    with open(file_path, "rb") as fp:
        b = pickle.load(fp)
    return b

def save_pickle(object, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)
    return

def my_mode(sample):
    c = Counter(sample)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]

def logging_func(log_file, message):
	with open(log_file,'a') as f:
		f.write(message)
	f.close()