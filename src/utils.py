import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# most of the code comes from https://github.com/ivishalanand/Federated-Learning-on-Hospital-Data/blob/master/Hospital%20data%20Federated%20learning.ipynb

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

def get_indexes_for_2_datasets(n, training = 80):
    indexes = get_random_indexes(n)
    train = int(training / 100. * n)
    return indexes[:train], indexes[train:]

def print_dataset(name, data):
    print('Dataset {}. Shape: {}'.format(name, data.shape))
    print(data)
    
def plot_metrics(test_losses, test_accs):
    assert len(test_losses) == len(test_accs)
    epochs = range(len(test_losses))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Test loss and accuracy')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    #ax1.set_xscale('log')
    ax1.set_ylabel("Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.plot(epochs, test_losses, label = "train")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ax1.plot(metrics["neurons"], metrics["min_val_loss"], label = "val")

    #ax2.set_xscale('log')
    ax2.set_ylabel("Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.plot(epochs, test_accs, label = "train");
    
def print_training_config():
    print("Not yet implemented")
    
def plot_some_samples(x, y = [], yhat = [], select_from = [], 
                      ncols = 6, nrows = 4, xdim = 28, ydim = 28,
                      label_mapping = range(100)):
    """
    Plot some input vectors as grayscale images (optionally together with their assigned or predicted labels).
    x is an NxD - dimensional array, where D is the length of an input vector and N is the number of samples.
    Out of the N samples, ncols x nrows indices are randomly selected from the list select_from (if it is empty, select_from becomes range(N)).
    """
    fig, ax = plt.subplots(nrows, ncols)
    if len(select_from) == 0:
        select_from = range(x.shape[0])
    indices = choice(select_from, size = min(ncols * nrows, len(select_from)), replace = False)
    for i, ind in enumerate(indices):
        thisax = ax[i//ncols,i%ncols]
        thisax.matshow(x[ind].reshape(xdim, ydim), cmap='gray')
        thisax.set_axis_off()
        if len(y) != 0:
            j = y[ind] if type(y[ind]) != np.ndarray else y[ind].argmax()
            thisax.text(0, 0, label_mapping[j], color='green', 
                                                       verticalalignment='top',
                                                       transform=thisax.transAxes)
        if len(yhat) != 0:
            k = yhat[ind] if type(yhat[ind]) != np.ndarray else yhat[ind].argmax()
            thisax.text(1, 0, label_mapping[k], color='red',
                                             verticalalignment='top',
                                             horizontalalignment='right',
                                             transform=thisax.transAxes)
    return fig
  