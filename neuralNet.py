# neuralNet.py
# Neural network implementation

import util
import classificationMethod
PRINT = True

class NeuralNetClassifier(classificationMethod.ClassificationMethod):
  """
  Neural Network classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels, iterations, layerSize):
    self.legalLabels = legalLabels
    self.iterations = iterations
    self.layerSize = layerSize
    self.type = "neuralnet"
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method." 
    pass

  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    Recall that a datum is a util.counter... 
    """
    pass
