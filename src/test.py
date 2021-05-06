from model import CNN
from server import Server
from constants import *

if __name__ == "__main__":
    server = Server(nr_clients=NR_CLIENTS, lr=LR, model=CNN(), epochs=CLIENT_EPOCHS, is_private=True)
    test_losses = []
    test_accs = []
    for nr_iter in range(NR_TRAINING_ITERATIONS):
        test_loss, test_acc = server.global_update()
        print(f"Global iteration {nr_iter + 1}, test loss: {test_loss}, test_acc: {test_acc}")
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    print(f"Test losses: {test_losses}")
    print(f"Test accuracies: {test_accs}")
    print("Finished")
