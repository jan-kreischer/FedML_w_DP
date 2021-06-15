import numpy as np
import torch
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# 1. Preprocessing utils
# source : https://github.com/ivishalanand/Federated-Learning-on-Hospital-Data/blob/master/Hospital%20data%20Federated%20learning.ipynb

def download_url(url, save_as):
    response = urllib.request.urlopen(url)
    data = response.read()
    file = open(save_as, 'wb')
    file.write(data)
    file.close()
    response.close()

def read_binary_file(file):
    f = open(file, 'rb')
    block = f.read()
    return block.decode('utf-16')

def split_text_in_lines(text):
    return text.split('\r\n')

def split_by_tabs(line):
    return line.split('\t')

def parse_double(field):
    field = field.replace(',', '.')
    return float(field)

def parse_boolean(field):
    return 1. if field == 'yes' else 0.

def read_np_array(file):
    text = read_binary_file(file)
    lines = split_text_in_lines(text)
    rows = []
    for line in lines:
        if line == '': continue
        line = line.replace('\r\n', '')
        fields = split_by_tabs(line)
        row = []
        j = 0
        for field in fields:
            value = parse_double(field) if j == 0 else parse_boolean(field)
            row.append(value)
            j += 1
        rows.append(row)
    matrix = np.array(rows, dtype=np.float32)
    return matrix

def get_random_indexes(n):
    indexes = list(range(n))
    random_indexes = []
    for i in range(n):
        r = np.random.randint(len(indexes))
        random_indexes.append(indexes.pop(r))
    return random_indexes

def get_indexes_for_2_datasets(n, training=80):
    indexes = get_random_indexes(n)
    train = int(training / 100. * n)
    return indexes[:train], indexes[train:]

def print_dataset(name, data):
    print('Dataset {}. Shape: {}'.format(name, data.shape))
    print(data)

# 2. Plot utils

def plot_exp(experiment_losses,experiment_accs,names,title):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for loss, label in zip(experiment_losses, names):
        ax1.plot(loss, label=label)
    for acc, label in zip(experiment_accs, names):
        ax2.plot(acc, label=label)
    ax1.set_ylabel('Global test loss')
    ax2.set_ylabel('Global test accuracy')
    ax2.set_xlabel('Global training rounds')
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    ax1.set_title(title)
    return fig
