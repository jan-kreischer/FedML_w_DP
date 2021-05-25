from server import Server
from constants import *
import warnings

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    server = Server(nr_clients=NR_CLIENTS,
                    lr=LR,
                    epochs=CLIENT_EPOCHS,
                    is_parallel=True,
                    is_private=False,
                    verbose="server")

    server()
