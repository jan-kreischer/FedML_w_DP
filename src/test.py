from model import CNN
from server import Server

NR_CLIENTS = 3
lr = 0.01
client_epochs = 15

if __name__ == "__main__":
    server = Server(nr_clients=NR_CLIENTS, lr=lr, model=CNN(), epochs=client_epochs)
    server.global_update()
    print("Finished")
