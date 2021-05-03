import torch
import torchvision
import syft as sy
from syft.util import get_root_data_path
from model import ConvNet, test_kwargs


class DataScientist:

    def __init__(self):
        self.model = None
        self.local_model = ConvNet(torch)
        self.test_loader = None
        self.test_kwargs = test_kwargs
        self.duet = None

    def load_test_data(self):
        # we need some transforms for the MNIST data set
        local_transform_1 = torchvision.transforms.ToTensor()  # this converts PIL images to Tensors
        local_transform_2 = torchvision.transforms.Normalize(0.1307, 0.3081)  # this normalizes the dataset

        # compose our transforms
        local_transforms = torchvision.transforms.Compose([local_transform_1, local_transform_2])

        test_data = torchvision.datasets.MNIST(str(get_root_data_path()), train=False, download=True,
                                               transform=local_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    def connect(self):
        self.duet = sy.join_duet(loopback=True)

    def send_model(self):
        self.model = self.local_model.send(self.duet)

    def test_local(self, torch_ref, test_loader, test_data_length):
        # download remote model
        if not self.model.is_local:
            local_model = self.model.get(
                request_block=True,
                reason="test evaluation",
                timeout_secs=5
            )
        else:
            local_model = self.model
        # + 0.5 lets us math.ceil without the import
        test_batches = round((test_data_length / args["test_batch_size"]) + 0.5)
        print(f"> Running test_local in {test_batches} batches")
        local_model.eval()
        test_loss = 0.0
        correct = 0.0

        with torch_ref.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                output = local_model(data)
                iter_loss = torch_ref.nn.functional.nll_loss(output, target, reduction="sum").item()
                test_loss = test_loss + iter_loss
                pred = output.argmax(dim=1)
                total = pred.eq(target).sum().item()
                correct += total

                if batch_idx >= test_batches - 1:
                    print("batch_idx >= test_batches, breaking")
                    break

        accuracy = correct / test_data_length
        print(f"Test Set Accuracy: {100 * accuracy}%")
