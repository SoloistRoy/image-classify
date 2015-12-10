# neuralNet.py
# Neural network implementation

import util
import classificationMethod
import random,math
from featureExtractor import *

PRINT = True

random.seed(0)

def rand(a, b):
    return (b - a) * random.random() + a


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


class NeuralNetClassifier(classificationMethod.ClassificationMethod):
    """
    Neural Network classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """




    def __init__(self, legalLabels, iterations, layerSize, type , eta = 0.25):
        self.legalLabels = legalLabels
        self.layerSize = layerSize
        self.times = iterations;


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

        self.hideSize = self.inputSize
        self.outSize = 10
        self.input_data = [0 for col in range(self.inputSize+1)]
        self.hide_data = [0 for col in range(self.hideSize+1)]
        self.out_data = [0 for col in range(self.outSize+1)]
        self.tar_data = [0 for col in range(self.outSize+1)]
        self.hid_delta = [0 for col in range(self.hideSize+1)]
        self.out_delta = [0 for col in range(self.outSize+1)]
        self.inp_hid_weight = [[self.randomizeWeight() for col in range(self.hideSize+1)] for row in range(self.inputSize+1)]
        self.hid_out_weight = [[self.randomizeWeight() for col in range(self.outSize+1)] for row in range(self.hideSize+1)]
        self.pre_inp_hid_weight = [[0 for col in range(self.hideSize+1)] for row in range(self.inputSize+1)]
        self.pre_hid_out_weight = [[0 for col in range(self.outSize+1)] for row in range(self.hideSize+1)]
        self.eta = eta
        self.move = 0.9

        self.ni = self.inputSize + 1
        self.nh = self.hideSize
        self.no = self.outSize
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        self.tar3 = [0];

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)


    def randomizeWeight(self):
        tmp = random.random()
        return tmp if util.flipCoin(0.5) else -tmp



    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."
        for t in range(self.times):
            print t
            for i in range(len(trainingData)):
                #trainingData[i].normalize()
                inputData = self.getArrayValueOfImg(trainingData[i])
                #if(trainingLabels[i] != 4):
                    #continue
                #label = self.normalizaLabel(trainingLabels[i])
                self.backProcTrain2(inputData,trainingLabels[i])
            self.vlab = validationLabels
            #for i in range(len(validationData)):
                #inputData = self.getArrayValueOfImg(validationData[i])
                #label = self.normalizaLabel(validationLabels[i])
                #self.backProcTrain(inputData,label)

    def getArrayValueOfImg(self, item):
        array = []
        for i in (range(self.ImgHeight)):
            out = ""
            for j in range(self.ImgWidth):
                out += str(item[(j,i)])
                array.append(item[(j,i)])
            #print out
        return array


    def backProcTrain(self, inputData, label):
        self.loadData(inputData)
        self.loadLabel(label)
        self.forecast()
        self.calculateDelta()
        self.updateWeight()

    def backProcTrain2(self, inputData, label):
        self.loadData2(inputData)
        self.loadLabel2(label)
        self.forecast2()
        self.calculateDelta2()
        self.updateWeight3()

    def backProcTrain3(self, inputData, label):
        self.loadData3(inputData)
        self.loadLabel3(label)
        self.forecast3()
        self.calculateDelta3()


    def forecast(self):
        for i in range(1,self.layerSize - 1):
            weight = self.weights[i-1]
            curData = self.data[i-1]
            nextData = self.data[i]
            for j in range(len(nextData)):
                sum = 0
                print curData
                for k in range(len(curData)):
                    sum += (weight[k][j] * curData[k])
                nextData[j] = self.sigmod(sum)
        sum = 0
        curData = self.data[-1]
        for k in range(len(self.predictionWeight)):
            #print self.predictionWeight[k]
            sum += (self.predictionWeight[k] * curData[k] )
        self.prediction = self.sigmod(sum)
        print ("prediction %f target %f" % (self.prediction,self.target))

    def forecast2(self):
        self.input_data[0] = 1
        for j in range(1,self.hideSize+1):
            sum = 0
            for i in range(self.inputSize+1):
                sum += self.inp_hid_weight[i][j] * self.input_data[i]
            self.hide_data[j] = self.sigmod(sum)

        self.hide_data[0] = 1
        for j in range(1,self.outSize+1):
            sum = 0
            for i in range(self.hideSize+1):
                sum += self.hid_out_weight[i][j] * self.hide_data[i]
            self.out_data[j] = self.sigmod(sum)


        print ("prediction %f target %f" % (self.getMax(self.out_data),self.getMax(self.tar_data)))

    def getMax(self,data):
        max = - 100000000
        index = -1
        for i in range(1,self.outSize+1):
            if(max < data[i]):
                max = data[i]
                index = i
        return index - 1

    def forecast3(self):
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = self.sigmod2(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = self.sigmod2(sum)

        print ("prediction %f target %f" % (self.ao[0],self.tar3[0]))

    def calculateDelta3(self):
        N = 0.5
        M = 0.1
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = self.tar3[k]-self.ao[k]
            output_deltas[k] = self.derivativeSigmoid2(self.ao[k]) * error

        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = self.derivativeSigmoid2(self.ah[j]) * error
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print(N*change, M*self.co[j][k])
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change



    def calculateDelta2(self):
        for i in range(self.outSize+1):
            o = self.out_data[i]
            self.out_delta[i] = self.derivativeSigmoid(o) * ( self.tar_data[i] - o)

        for j in range(self.hideSize+1):
            o = self.hide_data[j]
            sum = 0
            for k in range(1,self.outSize+1):
                sum += self.hid_out_weight[j][k] * self.out_delta[k]
            self.hid_delta[j] = self.derivativeSigmoid(o) * sum

    def calculateDelta(self):
        #print self.target - self.prediction;
        self.predictionDelta = self.derivativeSigmoid(self.prediction) * ( self.target  -  self.prediction )
        #print self.predictionDelta
        out = ""
        for i in range(len(self.deltas[-1])):
            sum = self.predictionWeight[i] * self.predictionDelta
            self.deltas[-1][i] = self.derivativeSigmoid(self.data[-1][i]) * sum
            out += str(self.deltas[-1][i])
        #print out
        if ((self.layerSize - 3) <= 0):
            return
        #print "test"
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




    def updateWeight3(self):
        self.hide_data[0] = 1
        for i in range(self.outSize + 1):
            for j in range(self.hideSize + 1):
                val = self.move * self.pre_hid_out_weight[j][i] + self.eta * self.out_delta[i] * self.hide_data[j]
                self.hid_out_weight[j][i] += val
                self.pre_hid_out_weight[j][i] = val

        self.input_data[0] = 1
        for i in range(self.hideSize + 1):
            for j in range(self.inputSize + 1):
                val = self.move * self.pre_inp_hid_weight[j][i] + self.eta * self.hid_delta[i] * self.input_data[j]
                self.inp_hid_weight[j][i] += val
                self.pre_inp_hid_weight[j][i] = val

    def updateWeight2(self):
        for l in range(self.layerSize - 2):
            curDel = self.deltas[l]
            curData = self.data[l]
            weight = self.weights[l]
            for i in range(len(curData)):
                for j in range(len(curDel)):
                    weight[i][j] =  weight[i][j] + (self.eta * curData[i] * curDel[j])
        out = ""
        for i in range(len(self.data[-1])):
            self.predictionWeight[i] +=  (self.eta * self.data[-1][i] * self.predictionDelta )
            out += "," + str(self.predictionWeight[i])
        #print out

    def updateWeight(self):
        for i in range(len(self.predictionWeight)):
            self.predictionWeight[i] = self.predictionWeight[i]  + (self.eta * self.data[-1][i] * self.predictionDelta )
        for l in reversed(range(self.layerSize-2)):
            curDel = self.deltas[l]
            curData = self.data[l]
            weight = self.weights[l]
            for i in range(len(curDel)):
                for j in range(len(curData)):
                    weight[j][i] = weight[j][i] + (self.eta * curData[j] * curDel[i])

    def sigmod2(self,val):
        return math.tanh(val)

    def sigmod(self,val):
        return 1.0/(1.0+math.exp(-val))

    def derivativeSigmoid(self,val):
        return self.sigmod(val)*(1.0 - self.sigmod(val))

    def derivativeSigmoid2(self,val):
        return 1.0 - val**2

    def loadData(self, inputData):
        self.data[0] = inputData

    def loadData2(self,inputData):
        self.input_data[0] = 0
        for i in range(self.inputSize):
            self.input_data[i+1] = inputData[i]

    def loadData3(self,inputData):
         for i in range(self.ni-1):
            self.ai[i] = inputData[i]

    def loadLabel(self, label):
        self.target = label

    def loadLabel2(self,lable):
        #print lable
        for i in range(self.outSize+1):
            self.tar_data[i] = 0
        self.tar_data[lable+1] = 1


    def loadLabel3(self,lable):
        self.tar3[0] = lable

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        result = []
        for i in range(len(data)):
            inputData = self.getArrayValueOfImg(data[i])
            result.append(self.getPrediction2(inputData,self.vlab[i]))
        return result

    def getPrediction3(self,inputData):
        self.loadData3(inputData)
        self.forecast3()
        return int(round(self.ao[0],1)*10) - 1

    def getPrediction2(self,inputData,labls):
        self.loadData2(inputData)
        self.loadLabel2(labls)
        self.forecast2()
        i = self.getMax(self.out_data)
        return i

    def getPrediction(self,inputData):
        self.loadData(inputData)
        self.forecast()
        return int(round(self.out_data[1],1)*10) - 1

    def normalizaLabel(self,val):
        return float((val) + 1) / 10.0

