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

    def __init__(self, model: nn.Module, data: DataLoader, len_data: int, lr: float, epochs: int, batch_size: int, n_accumulation_steps: int, epsilon_training_iteration: float, max_grad_norm: float, client_id: int,
                 loss=nn.NLLLoss(), is_private=False, verbose="all"):
        self.lr = lr
        self.model = model
        self.train_loader = data
        self.len_data = len_data
        self.epochs = epochs
        self.id = client_id
        self.criterion = loss
        self.is_private = is_private
        self.verbose = (verbose == "all" or verbose == "client")
        self.batch_size = batch_size
        self.n_accumulation_steps = n_accumulation_steps
        self.epsilon_training_iteration=epsilon_training_iteration
        self.max_grad_norm=max_grad_norm

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if is_private:
            self.delta = 1 / (2 * len_data)
            sample_rate = self.batch_size / self.len_data
            privacy_engine = PrivacyEngine(
                self.model,
                sample_rate=sample_rate * self.n_accumulation_steps,
                epochs=self.epochs,
                target_epsilon=self.epsilon_training_iteration,
                target_delta=self.delta,
                max_grad_norm=max_grad_norm,
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
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()

                if self.is_private:
                    # take a real optimizer step after N_VIRTUAL_STEP steps t
                    if ((i + 1) % self.n_accumulation_steps == 0) or ((i + 1) == self.len_data):
                        self.optimizer.step()
                    else:
                        # Take a virtual step
                        self.optimizer.virtual_step()
                else:
                    self.optimizer.step()

            if self.verbose:
                print(
                    f"[Client {self.id}]\t"
                    f"Train Epoch: {e}\t"
                    f"Loss: {loss.item():.4f}"
                )

            if self.is_private:
                epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(self.delta)
                if self.verbose: print(f"\t(ε = {epsilon:.2f}, δ = {self.delta}) for α = {best_alpha}")

        if self.verbose:
            print(f"Client {self.id} - done")
