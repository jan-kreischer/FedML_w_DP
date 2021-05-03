import torch.utils.data
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Dict


class FedMNIST:
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

    def get_client_data(self, client_id: int, batch_size: int = 64):
        return DataLoader(self.data_train_split[client_id], batch_size=batch_size, shuffle=True)
