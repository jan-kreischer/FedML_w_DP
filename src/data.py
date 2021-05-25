import torch.utils.data
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from typing import Dict
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from constants import BATCH_SIZE
from utils import download_url, read_np_array, get_indexes_for_2_datasets
import numpy as np

# TODO : reformating

class FedMNIST:
    """
    MNIST dataset, with samples randomly equally distributed among clients
    10 different classes (10 digits), images are 28x28
    """

    def __init__(self, nr_clients: int):
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
                              sample_rate=BATCH_SIZE / len(self.data_train_split[client_id]),
                          )), \
               len(self.data_train_split[client_id])


#class FEMNIST:
#    """
#    Federated Extended MNIST dataset
#    62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28x28
#    """
#    raise NotImplementedError

class FedMed:
    """
    Acute Inflammations dataset from the Center for Machine Learning and Intelligent Systems at University of California
    2 different classes, input has 6 attributes
    """
    def __init__(self, nr_clients: int, pred='inflammation'):
        names_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.names'
        data_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data'
        diagnosis_names = 'diagnosis.names'
        diagnosis_data = 'diagnosis.data'
        download_url(names_link, diagnosis_names)
        download_url(data_link, diagnosis_data)

        matrix = read_np_array(diagnosis_data)
        n_samples, n_dimensions = matrix.shape

        train_indexes, test_indexes = get_indexes_for_2_datasets(n_samples)

        data_train_mat = matrix[train_indexes]
        data_test_mat = matrix[test_indexes]

        data_train_x = torch.Tensor(data_train_mat[:, :6])
        data_test_x = torch.Tensor(data_test_mat[:, :6])

        if pred=='inflammation':
            data_train_y = torch.Tensor(np.vstack(data_train_mat[:, 6]))
            data_test_y = torch.Tensor(np.vstack(data_test_mat[:, 6]))
        elif pred=='nephritis':
            data_train_y = torch.Tensor(np.vstack(data_train_mat[:, 7]))
            data_test_y = torch.Tensor(np.vstack(data_test_mat[:, 7]))

        data_train = TensorDataset(data_train_x, data_train_y)
        self.data_test = TensorDataset(data_test_x, data_test_y)

        self.nr_clients = nr_clients
        len_train = len(data_train)
        self.len_client_data = {client_id: int(len_train / nr_clients) for client_id in range(nr_clients)}

        rs = random_split(data_train, list(self.len_client_data.values()))

        self.data_train_split: Dict[int, torch.utils.data.Subset] = {client_id: rs[client_id] for client_id in
                                                                     range(nr_clients)}

    def get_server_data(self):
        test_data = DataLoader(self.data_test, batch_size=BATCH_SIZE, shuffle=True)
        return test_data, self.len_client_data

    def get_client_data(self, client_id: int):
        return DataLoader(
            self.data_train_split[client_id],
                          batch_sampler=UniformWithReplacementSampler(
                              num_samples=len(self.data_train_split[client_id]),
                              sample_rate=BATCH_SIZE / len(self.data_train_split[client_id])
                          )
            ), len(self.data_train_split[client_id])
