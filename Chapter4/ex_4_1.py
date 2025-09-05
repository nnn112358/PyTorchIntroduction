""" このコードは関数シグネチャの例であり、実行は想定していません。
"""

class torchvision.datasets.MNIST(root, train=True, transform=None,
    target_transform=None, download=False)
class torchvision.datasets.CIFAR10(root, train=True, transform=None,
    target_transform=None, download=False)
class torchvision.datasets.VOCSegmentation(root, year='2012',
    image_set='train'， download=False, transform=None, 
    target_transform=None, transforms=None)
class torchvision.datasets.ImageNet(root, split='train', download=False,
    **kwargs)
class torchvision.datasets.CocoDetection(root, annFile, transform=None, 
    target_transform=None, transforms=None)
