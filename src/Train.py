#LIBRARIES
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR #LEARNING RATE'I EPICH'A GÖRE DEĞİŞTİRİYOR DÜŞÜRÜYOR

from timeit import default_timer
from tqdm.auto import tqdm

#SCRIPTS
from Datas import trainDataLoader,testDataLoader
from Model import getModel, device
from Helpers import accuracy, printTrainTime,trainStep,testStep,modelSummary

LEARNING_RATE = 0.01

model = getModel(numClasses=4,
                 device = device)

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr= LEARNING_RATE,
                            momentum=0.9) #Yani önceki gradyanın %90’ını hatırlayıp, %10’unu yeni gradyanla günceller.

scheduler = StepLR(optimizer=optimizer, #HANGİ OPTIMIZERI KULLNACAĞIMIZ
                   step_size=10, #KAÇ EPOCH'DA BİR LERANING RATE KÜÇÜLSÜN? BURDA 20
                   gamma=0.1) # LEARNING RATE'IN NE KADAR DÜŞECEĞİ BURDA 10 KAT DÜŞÜYOR 0.1 İLE

torch.manual_seed(28)
epochs = 60

#EARLY STOPPING AYARLARI
patience = 8
triggerTimes = 0
bestTestLoss = float("inf")

trainTimerStart = default_timer()

for epoch in tqdm(range(epochs)):
    trainStep(model=model,
              dataLoader=trainDataLoader,
              optimizer=optimizer,
              lossFn=lossFn,
              accFn=accuracy,
              device=device)
    
    testLoss = testStep(model=model,   #BURDA DEĞİKENE ATAMAYI UNUTMA RETURN ETTİĞİMİZ DEĞERİ
             dataLoader=testDataLoader,
             lossFn=lossFn,
             accFn=accuracy,
             device=device,
             returnLoss=True)
    
    #----------EARLY STOPPING KODLARI----------
    if testLoss < bestTestLoss: #TEST LOSS BEST TEST LOSS'dan küçük olursa modeli kaydediyor üzerine yazıyor yani
        bestTestLoss = testLoss
        triggerTimes = 0
        torch.save(model.state_dict(),"myCustomResNetModel.pth")
        print("AĞIRLIKLAR BAŞARIYLA KAYDEDİLDİ.")
    else: 
        triggerTimes += 1
        print(f"{triggerTimes} epoch'tur gelişme görülmedi.")
        if triggerTimes >= patience:
            print("Early Stopping triggered")
            break
    #----------EARLY STOPPING KODLARI----------

    scheduler.step()

trainTimerStop = default_timer()

printTrainTime(start=trainTimerStart,
               stop=trainTimerStop,
               device=device)

modelSum = modelSummary(model=model,
                        dataLoader=testDataLoader,
                        lossFn=lossFn,
                        accFn=accuracy,
                        device=device)
print(modelSum)


