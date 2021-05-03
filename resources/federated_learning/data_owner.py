# TorchVision hotfix https://github.com/pytorch/vision/issues/3549
from torchvision import datasets
from syft.util import get_root_data_path
import syft as sy

datasets.MNIST.resources = [
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    (
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
]
datasets.MNIST(str(get_root_data_path()), train=True, download=True)
datasets.MNIST(str(get_root_data_path()), train=False, download=True)


duet = sy.launch_duet(loopback=True)

print(duet.requests.pandas)

duet.requests.add_handler(
    action="accept"
)
