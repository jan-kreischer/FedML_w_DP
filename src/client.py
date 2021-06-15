import torch
from torch import nn, optim
import copy
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

class Client:
    """
    @param self:
    @param model:
    @param data: DataLoader enables to iteratively access a dataset in batches
    @param len_data
    @param lr:
    @param epochs:
    @param client_id:
    @param loss:
    @param is_private:
    @param verbose:
    @return:
    """

    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 data: DataLoader,
                 len_data: int,
                 epochs: int,
                 lr: float,
                 batch_size: int,
                 is_private: bool,
                 epsilon: float,
                 max_grad_norm: float,
                 noise_multiplier: float,
                 loss: nn.modules.loss,
                 device: torch.device,
                 verbose="all"):
        self.id = client_id
        self.model = model
        self.device = device
        self.train_loader = data
        self.len_data = len_data

        # Training hyper-parameters
        self.epochs = epochs
        self.criterion = loss
        self.lr = lr
        self.batch_size = batch_size

        # Differential Privacy
        self.is_private = is_private
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

        self.verbose = (verbose == "all" or verbose == "client")

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if is_private:
            self.delta = 1 / (2 * len_data)
            privacy_engine = PrivacyEngine(
                self.model,
                batch_size=self.batch_size,
                sample_size=self.len_data,
                epochs=self.epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=max_grad_norm
            )
            # Attach the privacy engine to the optimizer before running
            privacy_engine.attach(self.optimizer)

            if self.verbose:
                print(
                    f"[Client {self.id}]\t"
                    f"Using sigma={privacy_engine.noise_multiplier} and C={max_grad_norm}"
                )

    def receive_weights(self, model_params):
        """ Receive aggregated parameters, update model """
        self.model.load_state_dict(copy.deepcopy(model_params))

    def train(self):
        """ FedSGD algorithm, change local parameters
        Put model into training mode."""
        self.model.train()

        # Train the model for n epochs before aggregating
        for e in range(self.epochs):

            for i, (images, labels) in enumerate(self.train_loader):
                # Setting gradients to zero before starting backpropagation
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

            if self.verbose:
                print(
                    f"[Client {self.id}]\t"
                    f"Train Epoch: {e}\t"
                    f"Loss: {loss.item():.4f}"
                )

                if self.is_private:
                    epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(self.delta)
                    print(f"\t(ε = {epsilon:.2f}, δ = {self.delta}) for α = {best_alpha}")

        if self.verbose:
            print(f"Client {self.id} - done")
