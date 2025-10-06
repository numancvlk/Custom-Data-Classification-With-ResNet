#LIBRARIES
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import matplotlib.pyplot as plt


BATCH_SIZE = 64


dataTransforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

trainDatas = ImageFolder(
    root="DATAS\Train",
    transform= dataTransforms,
)

testDatas = ImageFolder(
    root="DATAS\Val",
    transform=dataTransforms
)

trainDataLoader = DataLoader(
    dataset=trainDatas,
    batch_size=BATCH_SIZE,
    shuffle=True
)

testDataLoader = DataLoader(
    dataset=testDatas,
    batch_size=BATCH_SIZE,
    shuffle=False
)
