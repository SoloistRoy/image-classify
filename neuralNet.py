# neuralNet.py
# Neural network implementation

import util
import classificationMethod
import random,math
from featureExtractor import *

PRINT = True


class NeuralNetClassifier(classificationMethod.ClassificationMethod):
    """
    Neural Network classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, iterations, layerSize, type , eta = 0.25):
        self.legalLabels = legalLabels
        self.iterations = iterations
        self.layerSize = layerSize
        self.times = iterations;
        self.eta = eta
        if(type == "faces"):
            self.ImgHeight = FACE_DATUM_HEIGHT
            self.ImgWidth = FACE_DATUM_WIDTH
            self.inputSize = self.ImgHeight * self.ImgWidth
        else:
            self.ImgHeight = DIGIT_DATUM_HEIGHT
            self.ImgWidth = DIGIT_DATUM_WIDTH
            self.inputSize = self.ImgHeight * self.ImgWidth
        self.data = [[0 for col in range(self.inputSize)] for row in range(self.layerSize - 1)]
        self.prediction = -1;

        self.weights = [[[self.randomizeWeight() for i in range(self.inputSize) ]  for col in range(self.inputSize)] for row in range(self.layerSize - 2)]
        self.preWeights = [[[0 for i in range(self.inputSize) ]  for col in range(self.inputSize)] for row in range(self.layerSize - 2)]
        self.predictionWeight = [self.randomizeWeight() for i in range(self.inputSize)]
        self.prePredictionWeight = [0 for i in range(self.inputSize)]

        self.deltas = [[0 for col in range(self.inputSize)] for row in range(self.layerSize - 2)]
        self.predictionDelta = 0;

        self.target = -1;

    def randomizeWeight(self):
        tmp = random.random()
        return tmp if util.flipCoin(0.5) else -tmp



    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."
        for t in range(self.times):
            for i in range(len(trainingData)):
                inputData = self.getArrayValueOfImg(trainingData[i])
                label = trainingLabels[i]
                self.backProcTrain(inputData,label)
            for i in range(len(validationData)):
                inputData = self.getArrayValueOfImg(validationData)
                label = validationLabels[i]
                self.backProcTrain(inputData,label)

    def getArrayValueOfImg(self, item):
        array = []
        for i in reversed(range(self.ImgHeight)):
            for j in range(self.ImgWidth):
                array.append(item[(j,i)])
        return array


    def backProcTrain(self, inputData, label):
        self.loadData(inputData)
        self.loadLabel(label)
        self.forecast()
        self.calculateDelta()
        self.updateWeight()

    def forecast(self):
        for i in range(1,self.layerSize - 1):
            weight = self.weights[i-1]
            curData = self.data[i-1]
            nextData = self.data[i]
            for j in range(len(nextData)):
                sum = 0
                for k in range(len(curData)):
                    sum += (weight[k][j] * curData[k])
                nextData[j] = self.sigmod(sum)
        sum = 0
        curData = self.data[-1]
        for k in range(len(self.predictionWeight)):
            sum += (self.predictionWeight[k] * curData[k] )
        self.prediction = self.sigmod(sum)

    def calculateDelta(self):
        self.predictionDelta = self.derivativeSigmoid(self.prediction) * (self.target - self.prediction)
        for i in range(len(self.deltas[-1])):
            sum = self.predictionWeight[i] * self.predictionDelta
            self.deltas[-1][i] = self.derivativeSigmoid(self.data[-1][i])*sum
        if ((self.layerSize - 3) <= 0):
            return
        for l in reversed(range(1,self.layerSize - 3)):
            weight = self.weights[l-1]
            curData = self.data[l]
            curDel = self.deltas[l-1]
            nextDel = self.deltas[l]
            for i in range(len(curDel)):
                sum = 0
                for j in range(len(nextDel)):
                    sum = weight[i][j] * nextDel[j]
                curDel = self.derivativeSigmoid(curData[i])*sum

    def updateWeight(self):
        for i in range(len(self.predictionWeight)):
            self.predictionWeight[i] +=  (self.eta * self.data[-1][i] * self.predictionDelta )
        for l in reversed(range(self.layerSize-2)):
            curDel = self.deltas[l]
            curData = self.data[l]
            weight = self.weights[l]
            for i in range(len(curDel)):
                for j in range(len(curData)):
                    weight[j][i] += (self.eta * curData[j] * curDel[i])

    def sigmod(self,val):
        return 1/(1+math.exp(-val))

    def derivativeSigmoid(self,val):
        return val*(1 - val)

    def loadData(self, inputData):
        self.data[0] = inputData

    def loadLabel(self, label):
        self.target = label

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        result = []
        for i in range(len(data)):
            inputData = self.getArrayValueOfImg(data[i])
            result.append(self.getPrediction(inputData[i]))
        return result

    def getPrediction(self,inputData):
        self.loadData(inputData)
        self.forecast()
        return self.prediction

