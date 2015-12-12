# neuralNet.py
# Neural network implementation

import util
import classificationMethod
import random, math
from featureExtractor import *

PRINT = True

class NeuralNetClassifier(classificationMethod.ClassificationMethod):
    """
    Neural Network classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, iterations, layerSize, type, eta=0.3):
        self.legalLabels = legalLabels
        self.layerSize = layerSize
        self.times = iterations
        if (type == "faces"):
            self.ImgHeight = FACE_DATUM_HEIGHT
            self.ImgWidth = FACE_DATUM_WIDTH
            self.inputSize = self.ImgHeight * self.ImgWidth
            self.max = 1
            self.min = 0
        else:
            self.ImgHeight = DIGIT_DATUM_HEIGHT
            self.ImgWidth = DIGIT_DATUM_WIDTH
            self.inputSize = self.ImgHeight * self.ImgWidth
            self.max = 2
            self.min = 0

        self.hideSize = self.inputSize
        self.outSize = 10
        self.input_data = [0 for col in range(self.inputSize + 1)]
        self.hide_data = [0 for col in range(self.hideSize + 1)]
        self.out_data = [0 for col in range(self.outSize + 1)]
        self.tar_data = [0 for col in range(self.outSize + 1)]
        self.hid_delta = [0 for col in range(self.hideSize + 1)]
        self.out_delta = [0 for col in range(self.outSize + 1)]
        self.inp_hid_weight = [[self.randomizeWeight() for col in range(self.hideSize + 1)] for row in
                               range(self.inputSize + 1)]
        self.hid_out_weight = [[self.randomizeWeight() for col in range(self.outSize + 1)] for row in
                               range(self.hideSize + 1)]
        self.pre_inp_hid_weight = [[0 for col in range(self.hideSize + 1)] for row in range(self.inputSize + 1)]
        self.pre_hid_out_weight = [[0 for col in range(self.outSize + 1)] for row in range(self.hideSize + 1)]
        self.eta = eta
        self.move = 1

    def randomizeWeight(self):
        tmp = random.random()
        return tmp if util.flipCoin(0.5) else -tmp

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."
        for t in range(self.times):
            print t
            indeces = range(len(trainingData))
            random.shuffle(indeces);
            for i in indeces:
                #self.normalizaInput(trainingData[i])
                inputData = self.getArrayValueOfImg(trainingData[i])
                self.backProcTrain(inputData, trainingLabels[i])
            self.vlab = validationLabels

    def getArrayValueOfImg(self, item):
        array = []
        for i in (range(self.ImgHeight)):
            out = ""
            for j in range(self.ImgWidth):
                out += str(item[(j, i)])
                array.append(item[(j, i)])
                # print out
        return array

    def backProcTrain(self, inputData, label):
        self.loadData(inputData)
        self.loadLabel(label)
        self.forecast()
        self.calculateDelta()
        self.updateWeight()

    def forecast(self):
        self.input_data[0] = 1
        for j in range(1, self.hideSize + 1):
            sum = 0
            for i in range(self.inputSize + 1):
                sum += self.inp_hid_weight[i][j] * self.input_data[i]
            #print sum
            self.hide_data[j] = self.sigmod(sum)

        self.hide_data[0] = 1
        for j in range(1, self.outSize + 1):
            sum = 0
            for i in range(self.hideSize + 1):
                sum += self.hid_out_weight[i][j] * self.hide_data[i]
            self.out_data[j] = self.sigmod(sum)

        print ("prediction %f target %f" % (self.getMax(self.out_data), self.getMax(self.tar_data)))

    def getMax(self, data):
        max = - 100000000
        index = -1
        for i in range(1, self.outSize + 1):
            if (max < data[i]):
                max = data[i]
                index = i
        return index - 1

    def calculateDelta(self):
        for i in range(self.outSize + 1):
            o = self.out_data[i]
            self.out_delta[i] = self.derivativeSigmoid(o) * (self.tar_data[i] - o)

        for j in range(self.hideSize + 1):
            o = self.hide_data[j]
            sum = 0
            for k in range(1, self.outSize + 1):
                sum += self.hid_out_weight[j][k] * self.out_delta[k]
            self.hid_delta[j] = self.derivativeSigmoid(o) * sum

    def updateWeight(self):
        self.hide_data[0] = 1
        for i in range(self.outSize + 1):
            for j in range(self.hideSize + 1):
                #val = self.move * self.pre_hid_out_weight[j][i] + self.eta * self.out_delta[i] * self.hide_data[j]
                #self.hid_out_weight[j][i] += val
                #self.pre_hid_out_weight[j][i] = val
                self.hid_out_weight[j][i] = self.hid_out_weight[j][i] * self.move + self.eta * self.out_delta[i] * self.hide_data[j]

        self.input_data[0] = 1
        for i in range(self.hideSize + 1):
            for j in range(self.inputSize + 1):
                #val = self.move * self.pre_inp_hid_weight[j][i] + self.eta * self.hid_delta[i] * self.input_data[j]
                #self.inp_hid_weight[j][i] += val
                #self.pre_inp_hid_weight[j][i] = val
                self.pre_inp_hid_weight[j][i] = self.pre_inp_hid_weight[j][i] * self.move + self.eta * self.hid_delta[i] * self.input_data[j]

    def sigmod(self, val):
        return 1.0 / (1.0 + math.exp(-val))

    def derivativeSigmoid(self, val):
        return self.sigmod(val) * (1.0 - self.sigmod(val))

    def loadData(self, inputData):
        self.input_data[0] = 0
        for i in range(self.inputSize):
            self.input_data[i + 1] = inputData[i]

    def loadLabel(self, lable):
        for i in range(self.outSize + 1):
            self.tar_data[i] = 0
        self.tar_data[lable + 1] = 1

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        result = []
        for i in range(len(data)):
            inputData = self.getArrayValueOfImg(data[i])
            result.append(self.getPrediction(inputData, self.vlab[i]))
        return result

    def getPrediction(self, inputData, labls):
        self.loadData(inputData)
        self.loadLabel(labls)
        self.forecast()
        i = self.getMax(self.out_data)
        return i

    def normalizaInput(self, data):
        for i in range(len(data)):
            data[i] = (data[i] - self.min) / (self.max - self.min)
