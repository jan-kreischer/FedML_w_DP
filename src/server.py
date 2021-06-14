import torch
from torch import nn
import copy
from client import Client
from typing import List, Dict
from data import FedMNIST, FEMNIST, FedMed
from opacus.dp_model_inspector import DPModelInspector
import threading
import numpy as np
from model import CNN, LogisticRegression


class Server:
    def __init__(self,
                 nr_clients: int,
                 nr_training_rounds: int,
                 data: str,
                 epochs: int,
                 lr: float,
                 batch_size: int,
                 is_private: bool,
                 epsilon: float,
                 max_grad_norm: float,
                 noise_multiplier: float,
                 is_parallel=False,
                 device=torch.device,
                 verbose="all"):

        self.config_summary({
            'nr_clients': nr_clients,
            'nr_training_rounds': nr_training_rounds,
            'data': data,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'is_private': is_private,
            'epsilon': epsilon,
            'max_grad_norm': max_grad_norm,
            'noise_multiplier': noise_multiplier,
            'is_parallel': is_parallel,
            'device': torch.device,
            'verbose': verbose
        })
        self.device = device
        self.nr_clients = nr_clients
        self.nr_training_rounds = nr_training_rounds
        self.data = data

        if self.data == 'MNIST':
            data_obj = FedMNIST(nr_clients=self.nr_clients, batch_size=batch_size, device=device)
            loss = nn.NLLLoss()
            model = CNN().to(device)
        elif self.data == 'FEMNIST':
            data_obj = FEMNIST(nr_clients=self.nr_clients, batch_size=batch_size, device=device)
            loss = nn.NLLLoss()
            model = CNN().to(torch.float)
            model = model.to(device)
        elif self.data == 'MED':
            data_obj = FedMed(nr_clients, batch_size=batch_size, device=device)
            loss = torch.nn.BCELoss(size_average=True)
            model = LogisticRegression().to(device)
        else:
            raise NotImplementedError(f'{self.data} is not implemented!')

        self.clients: Dict[int, Client] = {}
        inspector = DPModelInspector()
        assert inspector.validate(model), "The model is not valid"

        for i in range(self.nr_clients):
            data, len_data = data_obj.get_client_data(client_id=i)
            if epsilon:
                epsilon_round = epsilon / nr_training_rounds
            else:
                epsilon_round = None
            self.clients[i] = Client(
                client_id=i,
                model=copy.deepcopy(model),
                data=data,
                len_data=len_data,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                is_private=is_private,
                epsilon=epsilon_round,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                loss=loss,
                verbose=verbose,
            )

        self.test_data, self.clients_len_data = data_obj.get_server_data()
        self.len_train_data = np.sum(list(self.clients_len_data.values()))
        self.global_model = model
        self.criterion = loss
        self.is_parallel = is_parallel
        self.verbose = (verbose == "server" or verbose == "all")

    def aggregate(self, client_ids: List[int]):
        """ FedAvg algorithm, averaging all parameters """
        client_params = {client_id: self.clients[client_id].model.state_dict() for client_id in client_ids}
        new_params = copy.deepcopy(client_params[0])  # names
        for name in new_params:
            new_params[name] = torch.zeros(new_params[name].shape, device=self.device)
        for client_id, params in client_params.items():
            client_weight = self.clients_len_data[client_id] / self.len_train_data
            for name in new_params:
                new_params[name] += params[name] * client_weight  # averaging
        # set new parameters to global model
        self.global_model.load_state_dict(new_params)
        return self.global_model.state_dict()

    def broadcast_weights(self, new_params):
        """ Send to all clients """
        for client in self.clients.values():
            client.receive_weights(new_params.copy())

    def compute_acc(self):
        self.global_model.eval()

        test_loss = 0
        nr_correct = 0
        len_test_data = 0
        for attributes, labels in self.test_data:
            features = attributes.float()
            outputs = self.global_model(features)
            # accuracy
            if self.data == 'MNIST' or self.data == 'FEMNIST':
                pred_labels = torch.argmax(outputs, dim=1)
            elif self.data == 'MED':
                pred_labels = torch.round(outputs)
            else:
                raise NotImplementedError
            nr_correct += torch.eq(pred_labels, labels).type(torch.uint8).sum().item()
            len_test_data += len(attributes)
            # loss
            test_loss += self.criterion(outputs, labels)

        return nr_correct / len_test_data, test_loss.item()

    def global_update(self):
        # Permutation of the clients
        client_ids = np.random.choice(range(self.nr_clients), self.nr_clients, replace=False)

        if self.is_parallel:
            threads = []
            for client_id in client_ids:
                self.clients[client_id].model.share_memory()
                t = threading.Thread(target=Client.train, args=(self.clients[client_id],))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        else:
            for client_id in client_ids:
                self.clients[client_id].train()

        aggregated_weights = self.aggregate(list(self.clients.keys()))
        self.broadcast_weights(aggregated_weights)
        test_acc, test_loss = self.compute_acc()

        return test_loss, test_acc

    def train(self):
        print("--- Training ---")

        test_losses = []
        test_accs = []
        for training_round in range(self.nr_training_rounds):
            test_loss, test_acc = self.global_update()
            if self.verbose:
                print(f"Round {training_round + 1}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        print(f"Test losses: {list(np.around(np.array(test_losses), 4))}")
        print(f"Test accuracies: {test_accs}")
        print("Finished")
        return test_losses, test_accs

    @staticmethod
    def config_summary(config):
        print("--- Configuration ---")
        for key, value in config.items():
            print("{0}: {1}".format(key, value))
