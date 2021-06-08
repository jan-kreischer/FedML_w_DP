import torch
from torch import nn
import copy
from client import Client
from typing import List, Dict
from data import FedMNIST, FEMNIST, FedMed
from opacus.dp_model_inspector import DPModelInspector
import threading
import numpy as np
# from constants import DATA, NR_TRAINING_ROUNDS
from pytorchtools import EarlyStopping
from model import CNN, LogisticRegression


class Server:
    """
    @param self:
    @param nr_clients: The number of clients participating in the collaborative model creation
    @param lr:
    @param epochs: 
    @return:
    """

    '''
    def print_config_summary(self, nr_clients, nr_training_rounds, lr, epochs, data, batch_size, max_grad_norm, epsilon, n_accumulation_steps, epsilon_training_iteration, is_parallel, is_private):
    parallel_message = ''
    if is_parallel:
        parallel_message = 'in paralell'
    else 
        parallel_message = 'sequantially'
    print("Training {nr_clients} clients \n for {nr_training_round} training rounds \n on {data} dataset \n for {epochs} epochs with {batch_size} batch size and lr {lr} {parallel_message}".format(nr_clients=nr_clients, nr_training_rounds=nr_training_rounds, data=data, 
    epochs=epochs,batch_size=batch_size, lr=lr, parallel_message=parallel_message))
    '''

    def config_summary(self, config):
        print("--- Configuration ---")
        for key, value in config.items():
            print("{0}: {1}".format(key, value))

    def __init__(self,
                 nr_clients: int,
                 nr_training_rounds: int,
                 lr: float,
                 epochs: int,
                 data: str,
                 batch_size: int,
                 max_grad_norm: float,
                 epsilon: float,
                 n_accumulation_steps: int,
                 epsilon_training_iteration: float,
                 is_private=False,
                 is_parallel=False,
                 device=torch.device,
                 verbose="all"):

        self.config_summary({
            'nr_clients': nr_clients,
            'nr_training_rounds': nr_training_rounds,
            'lr': lr,
            'epochs': epochs,
            'data': data,
            'batch_size': batch_size,
            'max_grad_norm': max_grad_norm,
            'epsilon': epsilon,
            'n_accumulation_steps': n_accumulation_steps,
            'epsilon_training_iteration': epsilon_training_iteration,
            'is_parallel': is_parallel,
            'is_private': is_private,
            'device': torch.device,
            'verbose': verbose
        })

        self.nr_clients = nr_clients
        self.lr = lr
        self.nr_training_rounds = nr_training_rounds
        self.data = data

        if self.data == 'MNIST':
            print(self.nr_clients)
            data_obj = FedMNIST(nr_clients=self.nr_clients, batch_size=batch_size)
            loss = nn.NLLLoss()
            model = CNN()
        if self.data == 'FEMNIST':
            data_obj = FEMNIST(nr_clients=self.nr_clients, batch_size=batch_size, device=device)
            loss = nn.NLLLoss()
            model = CNN().double()
            model.to(device)
        elif self.data == 'Med':
            data_obj = FedMed(nr_clients, batch_size=batch_size)
            loss = torch.nn.BCELoss(size_average=True)
            model = LogisticRegression()
        else:
            raise NotImplementedError(f'{self.data} is not implemented!')

        self.clients: Dict[int, Client] = {}
        inspector = DPModelInspector()
        assert inspector.validate(model), "The model is not valid"

        for i in range(self.nr_clients):
            data, len_data = data_obj.get_client_data(client_id=i)
            self.clients[i] = Client(
                model=copy.deepcopy(model),
                data=data,
                len_data=len_data,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                n_accumulation_steps=n_accumulation_steps,
                epsilon_training_iteration=epsilon_training_iteration,
                max_grad_norm=max_grad_norm,
                client_id=i,
                loss=loss,
                is_private=is_private,
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
            new_params[name] = torch.zeros(new_params[name].shape)
        for client_id, params in client_params.items():
            client_weight = self.clients_len_data[client_id] / self.len_train_data
            for name in new_params:
                new_params[name] += params[name] * client_weight  # averaging
        # set new parameters to global model
        self.global_model.load_state_dict(copy.deepcopy(new_params))
        return self.global_model.state_dict().copy()

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
            features = attributes.double()
            outputs = self.global_model(features)
            # accuracy
            if self.data == 'MNIST':
                pred_labels = torch.argmax(outputs, dim=1)
            elif self.data == 'Med':
                pred_labels = torch.round(outputs)
            elif self.data == 'FEMNIST':
                pred_labels = torch.argmax(outputs, dim=1)
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

        #
        aggregated_weights = self.aggregate(list(self.clients.keys()))
        self.broadcast_weights(aggregated_weights)
        test_acc, test_loss = self.compute_acc()

        return test_loss, test_acc

    def __call__(self, early=False, patience=3, delta=0.05):
        print("--- Training ---")
        self.verbose = "all"
        if early:
            early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=self.verbose)

        test_losses = []
        test_accs = []
        for training_round in range(self.nr_training_rounds):
            test_loss, test_acc = self.global_update()
            if self.verbose: print(f"Round {training_round + 1}, test_loss: {test_loss:.4f}, test_acc: {test_acc}")
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            if early:
                early_stopping(test_loss, self.global_model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        # load last model if early
        if early: self.global_model.load_state_dict(torch.load('checkpoint.pt'))

        print(f"Test losses: {list(np.around(np.array(test_losses), 4))}")
        print(f"Test accuracies: {test_accs}")
        print("Finished")
        return test_losses, test_accs
