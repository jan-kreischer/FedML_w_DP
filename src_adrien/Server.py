import torch
from torch import nn, optim
import copy
from Client import *
from ML_model import *
import numpy as np

class Server(nn.Module):

    def __init__(self,nClient,lr,loaders,model,epochs):
        super(Server,self).__init__()
        self.nClient = nClient
        self.lr = lr

        self.clients = [Client(
            model=model,
            loader=loaders[i],
            lr=lr,
            epochs=epochs,
            n=i
        ) for i in range(self.nClient)]

        self.global_model = model()

    def aggregated(self, idxs):
        """ FedAvg algorithm, averaging all parameters """

        params = [self.clients[idx].model.state_dict() for idx in idxs]
        new_params = copy.deepcopy(params[0]) # names
        for name in new_params:
            new_params[name] = torch.zeros(new_params[name].shape)
        for idx, param in enumerate(params):
            for name in new_params:
                new_params[name] += params[name]/self.nClient # averaging

        # set new parameters to global model
        self.global_model.load_state_dict(copy.deepcopy(new_params))
        return self.global_model.load_state_dict().copy()

    def broadcast(self,new_params):
        """ Send to all clients """
        for client in self.clients:
            client.rcv(new_params.copy())

    def global_update(self):
        idxs = np.random.choice(range(self.nClient),self.nClient,replace=False) # permutation
        for idx in idxs:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs))

