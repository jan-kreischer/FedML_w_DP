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

def plot_exp(experiment_losses,experiment_accs,names):
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
    plt.show()

def plot_metrics(test_losses, test_accs):
    assert len(test_losses) == len(test_accs)
    epochs = range(len(test_losses))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Test loss and accuracy')
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    # ax1.set_xscale('log')
    ax1.set_ylabel("Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.plot(epochs, test_losses, label="train")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax1.plot(metrics["neurons"], metrics["min_val_loss"], label = "val")

    # ax2.set_xscale('log')
    ax2.set_ylabel("Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.plot(epochs, test_accs, label="train");


def plot_some_samples(x, y=[], yhat=[], select_from=[],
                      ncols=6, nrows=4, xdim=28, ydim=28,
                      label_mapping=range(100)):
    """
    Plot some input vectors as grayscale images (optionally together with their assigned or predicted labels).
    x is an NxD - dimensional array, where D is the length of an input vector and N is the number of samples.
    Out of the N samples, ncols x nrows indices are randomly selected from the list select_from (if it is empty, select_from becomes range(N)).
    """
    fig, ax = plt.subplots(nrows, ncols)
    if len(select_from) == 0:
        select_from = range(x.shape[0])
    indices = choice(select_from, size=min(ncols * nrows, len(select_from)), replace=False)
    for i, ind in enumerate(indices):
        thisax = ax[i // ncols, i % ncols]
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

# 3. Early stopping
# source : https://github.com/Bjarten/early-stopping-pytorch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='../data/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: '../data/checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
