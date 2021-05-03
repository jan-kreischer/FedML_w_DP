from torch import nn, optim
import copy
from torch.utils.data import DataLoader


class Client:
    def __init__(self, model: nn.Module, data: DataLoader, lr: float, epochs: int, client_id: int, loss=nn.NLLLoss()):
        self.lr = lr
        self.model = model
        self.train_loader = data
        self.epochs = epochs
        self.id = client_id
        self.criterion = loss

    def receive_weights(self, model_params):
        """ Receive aggregated parameters, update model """
        self.model.load_state_dict(copy.deepcopy(model_params))

    def train(self):
        """ FedSGD algorithm, change local parameters """
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        for e in range(self.epochs):
            for images, labels in self.train_loader:
                optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
            print(f"Client {self.id} - Epoch {e} done")

        print(f"Client {self.id} - done")
