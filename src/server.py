import torch
from torch import nn
import copy
from client import Client
import numpy as np
from typing import List, Dict
from data import FedMNIST
from torch.multiprocessing import Process
from opacus.dp_model_inspector import DPModelInspector


class Server:
    def __init__(self, nr_clients: int, lr: float, model: nn.Module, epochs: int, is_private=False):
        self.nr_clients = nr_clients
        self.lr = lr
        fedmnist_data = FedMNIST(nr_clients)
        self.clients: Dict[int, Client] = {}
        inspector = DPModelInspector()
        assert inspector.validate(model), "The model is not valid"
        for i in range(self.nr_clients):
            data, len_data = fedmnist_data.get_client_data(client_id=i)
            self.clients[i] = Client(
                model=copy.deepcopy(model),
                data=data,
                len_data=len_data,
                lr=lr,
                epochs=epochs,
                client_id=i,
                is_private=is_private
            )

        self.test_data, self.clients_len_data = fedmnist_data.get_server_data()
        self.len_train_data = np.sum(list(self.clients_len_data.values()))
        self.global_model = model
        self.criterion = nn.NLLLoss()

    def aggregate(self, client_ids: List[int]):
        """ FedAvg algorithm, averaging all parameters """
        client_params = {client_id: self.clients[client_id].model.state_dict() for client_id in client_ids}
        new_params = copy.deepcopy(client_params[0])  # names
        for name in new_params:
            new_params[name] = torch.zeros(new_params[name].shape)
        for client_id, params in client_params.items():
            client_weight = self.clients_len_data[client_id] / self.len_train_data
            for name in new_params:
                new_params[name] += params[name] / (client_weight / self.nr_clients)  # averaging
        # set new parameters to global model
        self.global_model.load_state_dict(copy.deepcopy(new_params))
        return self.global_model.state_dict().copy()

    def broadcast_weights(self, new_params):
        """ Send to all clients """
        for client in self.clients.values():
            client.receive_weights(new_params.copy())

    def compute_test_loss(self):
        self.global_model.eval()
        test_loss = 0
        for images, labels in self.test_data:
            pred = self.global_model(images)
            test_loss += self.criterion(pred, labels)

        return test_loss

    def compute_acc(self):
        self.global_model.eval()
        nr_correct = 0
        len_test_data = 0
        for images, labels in self.test_data:
            outputs = self.global_model(images)
            _, pred_labels = torch.max(outputs, 1)
            nr_correct += torch.eq(pred_labels, labels).type(torch.uint8).sum().item()
            len_test_data += len(images)

        return nr_correct / len_test_data

    def global_update(self):
        client_ids = np.random.choice(range(self.nr_clients), self.nr_clients, replace=False)  # permutation
        for client_id in client_ids:
            self.clients[client_id].train()

        # Issue: https://github.com/pytorch/pytorch/issues/35472
        # processes = [Process(target=client.train, args=()) for client in self.clients.values()]
        # for p in processes:
        #     p.start()
        # for p in processes:
        #     p.join()

        aggregated_weights = self.aggregate(list(self.clients.keys()))

        self.broadcast_weights(aggregated_weights)

        test_loss = self.compute_test_loss()
        test_acc = self.compute_acc()

        return test_loss, test_acc
