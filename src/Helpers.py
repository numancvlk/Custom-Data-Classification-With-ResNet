#LIBRARIES
import torch

#SCRIPTS
from Model import device


def accuracy(yTrue,yPred):
    correct = torch.eq(yTrue,yPred).sum().item()
    acc = (correct / len(yTrue)) * 100
    return acc

def printTrainTime(start,stop,device):
    totalTime = stop - start
    print(f"Total train time is {totalTime} on the {device}")

def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = device):
    
    trainLoss, trainAccuracy = 0,0

    model.train()

    for batch, (xTrain,yTrain) in  enumerate(dataLoader):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        #FORWARD
        trainPred = model(xTrain)

        #LOSS ACC
        loss = lossFn(trainPred,yTrain)
        trainLoss += loss.item()

        acc = accFn(yTrue = yTrain, yPred = trainPred.argmax(dim=1))
        trainAccuracy += acc

        #OPTIMIZER ZERO GRAD
        optimizer.zero_grad()

        #BACKWARD
        loss.backward()

        #STEP
        optimizer.step()

    
    trainLoss /= len(dataLoader)
    trainAccuracy /= len(dataLoader)

    print(f"TRAIN LOSS = {trainLoss:.5f} | TRAIN ACCURACY = {trainAccuracy:.2f}%")

def testStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = device,
              returnLoss = False):
    
    testLoss, testAccuracy = 0,0
    
    model.eval()
    with torch.inference_mode():
        for xTest,yTest in dataLoader:
            xTest,yTest = xTest.to(device), yTest.to(device)

        
            testPred = model(xTest)

            loss = lossFn(testPred, yTest)
            testLoss += loss.item()

            acc = accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
            testAccuracy += acc
    
    testLoss /= len(dataLoader)
    testAccuracy /= len(dataLoader)
    print(f"TEST LOSS = {testLoss:.5f} | TEST ACCURACY = {testAccuracy:.2f}%")

    if returnLoss: #EARLY STOPPING İÇİN EKLEDİĞİMİZ BİR ŞEY
        return testLoss


def modelSummary(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = device):
    
    summaryLoss, summaryAccuracy = 0,0
    
    model.eval()
    with torch.inference_mode():
        for xTest,yTest in dataLoader:
            xTest,yTest = xTest.to(device), yTest.to(device)

        
            testPred = model(xTest)

            loss = lossFn(testPred, yTest)
            summaryLoss += loss.item()

            acc = accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
            summaryAccuracy += acc

    summaryLoss /= len(dataLoader)
    summaryAccuracy /= len(dataLoader)

    return {"MODEL NAME": model.__class__.__name__,
            "MODEL LOSS": summaryLoss,
            "MODEL ACCURACY": summaryAccuracy}