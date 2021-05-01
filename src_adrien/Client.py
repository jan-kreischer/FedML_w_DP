import torch
from torch import nn, optim
import copy
from ML_model import *

class Client(nn.Module):

    def __init__(self,model,loader,lr,epochs,n):
        super(Client,self).__init__()
        self.lr = lr
        self.model = model()
        self.train_loader = loader
        self.epochs = epochs
        self.n = n

    def rcv(self,model_param):
        """ Receive aggregated parameters, update model """
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """ FedSGD algorithm, change local parameters """
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(),lr=self.lr)

        for e in range(self.epochs):
            for images,labels in self.train_loader:
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            print("Client {} - Epoch {} done".format(self.n,e))

        print("Client {} - done".format(self.n))



