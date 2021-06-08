import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from typing import Dict
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from utils import download_url, read_np_array, get_indexes_for_2_datasets
import numpy as np
import pandas as pd


class FedMNIST:
    """
    MNIST dataset, with samples randomly equally distributed among clients
    10 different classes (10 digits), images are 28x28
    """

    def __init__(self, nr_clients: int, batch_size: int):
        self.batch_size = batch_size

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        try:
            data_train = datasets.MNIST(root="./data",
                                        train=True,
                                        download=False,
                                        transform=transform)
            self.data_test = datasets.MNIST(root="./data",
                                            train=False,
                                            download=False,
                                            transform=transform)
        except RuntimeError:
            data_train = datasets.MNIST(root="./data",
                                        train=True,
                                        download=True,
                                        transform=transform)
            self.data_test = datasets.MNIST(root="./data",
                                            train=False,
                                            download=True,
                                            transform=transform)

        self.nr_clients = nr_clients
        len_train = len(data_train)
        self.len_client_data = {client_id: int(len_train / nr_clients) for client_id in range(nr_clients)}
        rs = random_split(data_train, list(self.len_client_data.values()))

        self.data_train_split: Dict[int, torch.utils.data.Subset] = {client_id: rs[client_id] for client_id in
                                                                     range(nr_clients)}

    def get_server_data(self, batch_size: int = 64):
        test_data = DataLoader(self.data_test, batch_size=batch_size, shuffle=True)

        return test_data, self.len_client_data

    def get_client_data(self, client_id: int):
        return DataLoader(self.data_train_split[client_id],
                          batch_sampler=UniformWithReplacementSampler(
                              num_samples=len(self.data_train_split[client_id]),
                              sample_rate=self.batch_size / len(self.data_train_split[client_id]),
                          )), \
               len(self.data_train_split[client_id])


# TODO : reformatting


class FEMNIST:
    """
    MNIST dataset, with samples randomly equally distributed among clients
    10 different classes (10 digits), images are 28x28
    """

    def chunks(self, X, y, size):
        return ((X[i::size], y[i::size]) for i in range(size))

    def makeArray(self, string):
        return np.fromstring(string[1:-1], sep=', ', dtype=np.double)

    def __init__(self, nr_clients: int, batch_size: int, device: torch.device):
        self.batch_size = batch_size
        # --- Load Test Data ---
        print("--- Load Data ---")
        data_path = './data/FEMNIST/test.csv'
        data = pd.read_csv(data_path, dtype=[('id', np.double), ('X', str), ('y', int)])
        # Convert string encoded X into numpy array
        data['X'] = data['X'].apply(lambda string: np.fromstring(string[1:-1], sep=', ', dtype=np.double))
        # Stack all numpy arrays on top of each other
        Xs = np.vstack(data['X'])
        ys = data['y'].to_numpy()

        self.data_test = TensorDataset(torch.reshape(torch.tensor(Xs, device=device, dtype=torch.float64), (-1, 1, 28, 28)),
                                       torch.tensor(ys, device=device, dtype=torch.int64))
        print("Loaded Test Data")

        # --- Load Training Data ---
        self.data_train_split = {}
        self.len_client_data = {}
        data_path = './data/FEMNIST/train.csv'

        data = pd.read_csv(data_path, dtype=[('id', np.double), ('X', str), ('y', int)])
        # Convert string encoded X into numpy array
        data['X'] = data['X'].apply(lambda string: np.fromstring(string[1:-1], sep=', ', dtype=np.double))
        # Stack all numpy arrays on top of each other
        Xs = np.vstack(data['X'])
        ys = data['y'].to_numpy()

        client_id = 0
        for X_chunk, y_chunk in self.chunks(Xs, ys, nr_clients):
            self.len_client_data[client_id] = len(X_chunk)
            client_dataset = TensorDataset(torch.reshape(torch.tensor(X_chunk, device=device, dtype=torch.float64), (-1, 1, 28, 28)),
                                           torch.tensor(y_chunk, device=device, dtype=torch.int64))
            client_subset = torch.utils.data.Subset(client_dataset, indices=np.arange(len(client_dataset)))
            self.data_train_split[client_id] = client_subset
            client_id += 1
        print("Loaded Training Data")

    def get_server_data(self, batch_size: int = 64):
        test_data = DataLoader(self.data_test, batch_size=batch_size, shuffle=True)
        return test_data, self.len_client_data

    def get_client_data(self, client_id: int):
        return DataLoader(self.data_train_split[client_id],
                          batch_sampler=UniformWithReplacementSampler(
                              num_samples=len(self.data_train_split[client_id]),
                              sample_rate=self.batch_size / len(self.data_train_split[client_id]),
                          )), \
               len(self.data_train_split[client_id])


class FedMed:
    """
    Acute Inflammations dataset from the Center for Machine Learning and Intelligent Systems at University of California
    2 different classes, input has 6 attributes
    """

    def __init__(self, nr_clients: int, batch_size: int, pred='inflammation'):
        self.batch_size = batch_size

        names_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.names'
        data_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data'
        diagnosis_names = './data/MED/diagnosis.names'
        diagnosis_data = './data/MED/diagnosis.data'
        download_url(names_link, diagnosis_names)
        download_url(data_link, diagnosis_data)

        matrix = read_np_array(diagnosis_data)
        n_samples, n_dimensions = matrix.shape

        train_indexes, test_indexes = get_indexes_for_2_datasets(n_samples)

        data_train_mat = matrix[train_indexes]
        data_test_mat = matrix[test_indexes]

        data_train_x = torch.Tensor(data_train_mat[:, :6])
        data_test_x = torch.Tensor(data_test_mat[:, :6])

        if pred == 'inflammation':
            data_train_y = torch.Tensor(np.vstack(data_train_mat[:, 6]))
            data_test_y = torch.Tensor(np.vstack(data_test_mat[:, 6]))
        elif pred == 'nephritis':
            data_train_y = torch.Tensor(np.vstack(data_train_mat[:, 7]))
            data_test_y = torch.Tensor(np.vstack(data_test_mat[:, 7]))
        else:
            raise ValueError(f"Feature {pred} does not exist!")

        data_train = TensorDataset(data_train_x, data_train_y)
        self.data_test = TensorDataset(data_test_x, data_test_y)

        self.nr_clients = nr_clients
        len_train = len(data_train)
        self.len_client_data = {client_id: int(len_train / nr_clients) for client_id in range(nr_clients)}

        rs = random_split(data_train, list(self.len_client_data.values()))

        self.data_train_split: Dict[int, torch.utils.data.Subset] = {client_id: rs[client_id] for client_id in
                                                                     range(nr_clients)}

    def get_server_data(self):
        test_data = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=True)
        return test_data, self.len_client_data

    def get_client_data(self, client_id: int):
        return DataLoader(
            self.data_train_split[client_id],
            batch_sampler=UniformWithReplacementSampler(
                num_samples=len(self.data_train_split[client_id]),
                sample_rate=self.batch_size / len(self.data_train_split[client_id])
            )
        ), len(self.data_train_split[client_id])
