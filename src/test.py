from model import CNN, LogisticRegression
from server import Server
from constants import *
import numpy as np
import warnings

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    server = Server(nr_clients=NR_CLIENTS, lr=LR, model=LogisticRegression(), epochs=CLIENT_EPOCHS, is_parallel=True,
                    is_private=True)
    test_losses = []
    test_accs = []
    for training_round in range(NR_TRAINING_ROUNDS):
        test_loss, test_acc = server.global_update()
        print(f"Round {training_round + 1}, test_loss: {test_loss:.4f}, test_acc: {test_acc}")
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    print(f"Test losses: {list(np.around(np.array(test_losses), 4))}")
    print(f"Test accuracies: {test_accs}")
    print("Finished")
