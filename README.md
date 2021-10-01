# Federated Learning with PyTorch

This repository is a simple PyTorch implementation of **Cross-Silo Federated Learning (FL)** with **Differential Privacy (DP)**.  
The server contains a list of client instances that perform the training on local data. The Federated Averaging (FedAvg)
method is used to combine the updated client models in the global model.  
Each client applies a DP mechanism locally using [Opacus](https://opacus.ai/) to perturb trained parameters before uploading to the parameter server.  

Project Contributors
- Adrien Banse
- Xavi Oliva
- Jan Bauer

## Files (src)

> client.py: definition of the FL client class

> server.py: definition of the FL server class

> model.py: CNN PyTorch model

> data.py: definition of data loaders

> utils.py: definition of auxiliary functions

## Files (experiments)

> exp_FedMNIST.ipynb: notebook, experiments for the MNIST database

> exp_FEMNIST.ipynb: notebook, experiments for the FEMMNIST database

> exp_FedMed.ipynb: notebook, experiments for the medical database

## Usage
1. Install requirements with ```pip install -r requirements.txt```
2. Ensure to put training data into the ```data``` directory.
3. Run any notebook in the experiments directory

### Federated Learning Parameters

> **NR_CLIENTS**:
*Number of distributed clients participating in the training process.*<br/>
> **NR_TRAINING_ROUNDS**:
*Number of times that the server performs the global model update<br/> 
based on the weights collected from the local client models.*<br/>
> **DATA**:
*Chosen data (MNIST, FEMNIST or MED).*<br/>
> **EPOCHS**:
*Number of epochs that each client is trained during one global training round.*<br/>
> **LR**:
*Learning rate used for sgd.*<br/>
> **BATCH_SIZE**:
*Size of the training batch.*<br/>

### Differential Privacy Parameters

>**EPSILON**:
*Privacy loss parameter, determining how much noise is added to the computation.*<br/>
>**MAX_GRAD_NORM**:
*The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.*<br/>
>**NOISE_MULTIPLIER**:
*Noise multiplier (during the addition of noise).*<br/>

### Other parameters

>**IS_PRIVATE**:
*Activate privacy.*<br/>
>**IS_PARALLEL**:
*Activate parallelization during clients training.*<br/>
>**DEVICE**:
*TORCH device.*<br/>
>**VERBOSE**:
*Verbose parameter (server, client, all or other).*<br/>
