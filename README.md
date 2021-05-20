# Federated Learning with PyTorch

This repository is a simple PyTorch implementation of **Cross-Silo Federated Learning (FL)** with **Differential Privacy (DP)**.
The server contains a list of client instances that perform the training on local data. The Federated Averaging (FedAvg)
method is used to combine the updated client models in the global model.
Each client applies a DP mechanism locally using [Opacus](https://opacus.ai/) to perturb trained parameters before uploading to the parameter server.

## Files

> client.py: definition of the FL client class

> server.py: definition of the FL server class

> model.py: CNN PyTorch model

> data.py: definition of data loaders

## Usage
1. Install requirements with ```pip install -r requirements.txt```
2. Set parameters in ```constants.py```
3. Run ```python test.py```

### Federated Learning Parameters

> NR_CLIENTS: <br>
> LR: <br>
> CLIENT_EPOCHS: <br>
> NR_TRAINING_ROUNDS

### Differential Privacy Parameters

> MAX_GRAD_NORM: <br>
> EPSILON: <br>
> DELTA: <br>
> VIRTUAL_BATCH_SIZE: <br>
> N_ACCUMULATION_STEPS: <br>
