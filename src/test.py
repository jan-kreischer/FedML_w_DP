from model import CNN, LogisticRegression
from server import Server
from constants import *
import numpy as np
import warnings
from pytorchtools import EarlyStopping
import torch

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    server = Server(nr_clients=NR_CLIENTS, lr=LR, model=LogisticRegression(), epochs=CLIENT_EPOCHS, is_parallel=True,
                    is_private=False, verbose=False)

    early_stopping = EarlyStopping(patience=3, delta=0.05, verbose=True)

    test_losses = []
    test_accs = []
    for training_round in range(NR_TRAINING_ROUNDS):
        test_loss, test_acc = server.global_update()
        print(f"Round {training_round + 1}, test_loss: {test_loss:.4f}, test_acc: {test_acc}")
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        early_stopping(test_loss, server.global_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load last model
    server.global_model.load_state_dict(torch.load('checkpoint.pt'))

    print(f"Test losses: {list(np.around(np.array(test_losses), 4))}")
    print(f"Test accuracies: {test_accs}")
    print("Finished")
