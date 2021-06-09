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

> utils.py: definition of auxiliary functions

## Usage
1. Install requirements with ```pip install -r requirements.txt```
2. Ensure to put training data into the ```src/data``` directory.
3. Run ```index.ipynb```

### Federated Learning Parameters

> **NR_CLIENTS**: <br/>
*Number of distributed clients participating in the training process.*<br/>
> **LR**: <br/>
*Learning rate used for sgd.*<br/>
> **NR_TRAINING_ROUNDS**:<br/>
*Number of times that the server performs the global model update<br/> 
based on the weights collected from the local client models.*<br/>
> **CLIENT_EPOCHS**: <br/>
*Number of epochs that each client is trained during one global training round.*<br/>

### Differential Privacy Parameters

>**MAX_GRAD_NORM**:<br/>
*The maximum L2 norm of per-sample gradients before they are aggregated by the averaging step.*<br/>
>**EPSILON**:<br/>
*Privacy loss parameter, determining how much noise is added to the computation.*<br/>
>**DELTA**:<br>
*The target δ of the (ϵ,δ)-differential privacy guarantee.*<br/>
>**VIRTUAL_BATCH_SIZE**:<br/>
*The average of n mini batches is accumulated into one virtual step of this size in order to save memory.*<br/>
>**N_ACCUMULATION_STEPS**:<br/>
*The number of times a normal sgd step has to be performed in order to be able to perform one virtual step.*<br/>

