import torch as th
from torchvision import datapisets, transforms
from opacus import PrivacyEngine
import syft as sy

hook = sy.TorchHook(th)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
workers = [alice, bob]

sy.local_worker.is_client_worker = False

train_datasets = datasets.MNIST('../mnist',
                                train=True, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,)), ])
                                ).federate(*workers)


def make_model():
    return th.nn.Sequential(
        th.nn.Conv2d(1, 16, 8, 2, padding=3),
        th.nn.ReLU(),
        th.nn.MaxPool2d(2, 1),
        th.nn.Conv2d(16, 32, 4, 2),
        th.nn.ReLU(),
        th.nn.MaxPool2d(2, 1),
        th.nn.Flatten(),
        th.nn.Linear(32 * 4 * 4, 32),
        th.nn.ReLU(),
        th.nn.Linear(32, 10)
    )


# the local version that we will use to do the aggregation
local_model = make_model()

models, dataloaders, optimizers, privacy_engines = [], [], [], []
for worker in workers:
    model = make_model()
    optimizer = th.optim.SGD(model.parameters(), lr=0.1)
    model.send(worker)
    dataset = train_datasets[worker.id]
    dataloader = th.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    privacy_engine = PrivacyEngine(model,
                                   batch_size=128,
                                   sample_size=len(dataset),
                                   alphas=range(2, 32),
                                   noise_multiplier=1.2,
                                   max_grad_norm=1.0)
    privacy_engine.attach(optimizer)

    models.append(model)
    dataloaders.append(dataloader)
    optimizers.append(optimizer)
    privacy_engines.append(privacy_engine)


def send_new_models(local_model, models):
    with th.no_grad():
        for remote_model in models:
            for new_param, remote_param in zip(local_model.parameters(), remote_model.parameters()):
                worker = remote_param.location
                remote_value = new_param.send(worker)
                remote_param.set_(remote_value)


def federated_aggregation(local_model, models):
    with th.no_grad():
        for local_param, *remote_params in zip(
                *([local_model.parameters()] + [model.parameters() for model in models])):
            param_stack = th.zeros(*remote_params[0].shape)
            for remote_param in remote_params:
                param_stack += remote_param.copy().get()
            param_stack /= len(remote_params)
            local_param.set_(param_stack)


def train(epoch, delta):
    # 1. Send new version of the model
    send_new_models(local_model, models)

    # 2. Train remotely the models
    for i, worker in enumerate(workers):
        dataloader = dataloaders[i]
        model = models[i]
        optimizer = optimizers[i]

        model.train()
        criterion = th.nn.CrossEntropyLoss()
        losses = []
        for i, (data, target) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.get().item())

        sy.local_worker.clear_objects()
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print(
            f"[{worker.id}]\t"
            f"Train Epoch: {epoch} \t"
            f"Loss: {sum(losses) / len(losses):.4f} "
            f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")

    # 3. Federated aggregation of the updated models
    federated_aggregation(local_model, models)


for epoch in range(5):
    train(epoch, delta=1e-5)
