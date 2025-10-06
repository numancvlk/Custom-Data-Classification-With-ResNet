#LIBRARIES
import torch
from torch import nn
from torchvision.models import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

def getModel(numClasses = 3, device = "cpu"):
    model = resnet18(pretrained = True) #RESNET 18 ' I DAHA ÖNCE EĞİTİLMİŞ AĞIRLIKLARI İLE ÇAĞIRDIK
    
    #MODELİN GÖVDESİNİ DONDURUYORUZ BURADA BURDAKİ AĞIRLIKLAR DEĞİŞMEYECEK
    for param in model.parameters():
        param.requires_grad = False

    numFeatures = model.fc.in_features #SON KATMANA GİREN BİLGİ MİKTARINI GÖSTERİR RESNET 18 DE 512 SANIRIM

    model.fc = nn.Linear(in_features=numFeatures,
                         out_features=numClasses)
    
    model.to(device)
    return model