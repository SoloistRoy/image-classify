# dataClassifier.py
# This file contains harness code for data classification

import mostFrequent
import perceptron
import naiveBayes
import neuralNet
import samples
import util
import sys

from featureExtractor import *


def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.
    Use the printImage(<list of pixels>) function to visualize features.
    An example of use has been given to you.

    - classifier is the trained classifier
    - guesses is the list of labels predicted by your classifier on the test set
    - testLabels is the list of true labels
    - testData is the list of training datapoints (as util.Counter of features)
    - rawTestData is the list of training datapoints (as samples.Datum)
    - printImage is a method to visualize the features 
    (see its use in the odds ratio part in runClassifier method)

    This code won't be evaluated. It is for your own optional use (and you can
    modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    for i in range(len(guesses)):
        prediction = guesses[i]
        truth = testLabels[i]
        if (prediction != truth):
            print "==================================="
            print "Mistake on example %d" % i 
            print "Predicted %d; truth is %d" % (prediction, truth)
            print "Image: "
            print rawTestData[i]
            break


#===============================================#
# You don't have to modify any code below.
#===============================================#


class ImagePrinter:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def printImage(self, pixels):
        """
        Prints a Datum object that contains all pixels in the provided list of
        pixels. This will serve as a helper function to the analysis function
        you write.

        Pixels should take the form [(2,2), (2, 3), ...] where each tuple
        represents a pixel.
        """
        image = samples.Datum(None,self.width,self.height)
        for pix in pixels:
            try:
                # This is so that new features that you could define which are
                # not of the form of (x,y) will not break this image printer...
                x,y = pix
                image.pixels[x][y] = 2
            except:
                print "new features:", pix
                continue
        print image    

def default(str):
    return str + ' [Default: %default]'

MF, PT, NB, NN = 0, 1, 2, 3
CLASSIFIER = {
    MF: 'Most Frequent',
    PT: 'Perception',
    NB: 'Naive Bayes',
    NN: 'Neural Network'
}

def readCommand( argv ):
    "Processes the command used to run from the command line."
    from optparse import OptionParser    
    parser = OptionParser(USAGE_STRING)

    parser.add_option('-c', '--classifier', help=default('Classifier=' + str(CLASSIFIER)), default=0, type="int")
    parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
    parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
    parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
    parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
    parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
    parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
    parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
    parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
    parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
    parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
    parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-l', '--layerSize', help=default('Layer Size for Neural Network'), default=3, type="int")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
    args = {}

    # Set up variables according to the command line input.
    print "Doing classification"
    print "--------------------------------------------------"
    print "data:\t\t\t\t" + options.data
    print "classifier:\t\t\t" + CLASSIFIER[options.classifier]
    print "using enhanced features:\t" + str(options.features)
    print "training set size:\t\t" + str(options.training)

    if(options.data=="digits"):
        printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorDigit
        else:
            featureFunction = basicFeatureExtractorDigit
        legalLabels = range(10)
    elif(options.data=="faces"):
        printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
        if (options.features):
            featureFunction = enhancedFeatureExtractorFace
        else:
            featureFunction = basicFeatureExtractorFace            
        legalLabels = range(2)
    else:
        print "Unknown dataset", options.data
        print USAGE_STRING
        sys.exit(2)

    if options.training <= 0:
        print "Training set size should be a positive integer (you provided: %d)" % options.training
        print USAGE_STRING
        sys.exit(2)

    if options.smoothing <= 0:
        print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
        print USAGE_STRING
        sys.exit(2)

    if options.odds:
        if options.label1 not in legalLabels or options.label2 not in legalLabels:
            print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
            print USAGE_STRING
            sys.exit(2)

    if(options.classifier == MF):
        classifier = mostFrequent.MostFrequentClassifier(legalLabels)
    elif(options.classifier == PT):
        classifier = perceptron.PerceptronClassifier(legalLabels, options.iterations)
        print "iterations:\t\t\t" + str(options.iterations)
    elif(options.classifier == NB):
        classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
        classifier.setSmoothing(options.smoothing)
        if (options.autotune):
            print "using automatic tuning:\t\tTrue"
            classifier.automaticTuning = True
        else:
            print "using smoothing parameter k=%f for naivebayes" % options.smoothing
    elif(options.classifier == NN):
        classifier = neuralNet.NeuralNetClassifier(legalLabels, options.iterations, options.layerSize)
        print "iterations:\t\t\t" + str(options.iterations)
        print "layer size:\t\t\t" + str(options.layerSize)
        print "using automatic tuning:\t\t" + str(options.autotune)
        if (options.autotune):
            # TODO
            classifier.automaticTuning = True
        else:
            print "using default parameter:\teta=0.25"
    else:
        print "Unknown classifier:", options.classifier
        print USAGE_STRING
        sys.exit(2)

    print "--------------------------------------------------"

    args['classifier'] = classifier
    args['featureFunction'] = featureFunction
    args['printImage'] = printImage

    return args, options

USAGE_STRING = """
    USAGE: python dataClassifier.py <options>
    EXAMPLES: 
    (1) python dataClassifier.py
     - trains the default mostFrequent classifier on the digit dataset using
       the default 100 training examples and then test the classifier on test
       data
    (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
     - would run the naive Bayes classifier on 1000 training examples using the
       enhancedFeatureExtractorDigits function to get the features on the faces
       dataset, would use the smoothing parameter equals to 2.5, would test the
       classifier on the test data and performs an odd ratio analysis with
       label1=3 vs.  label2=6 
"""

# Main harness code

def runClassifier(args, options):

    featureFunction = args['featureFunction']
    classifier = args['classifier']
    printImage = args['printImage']

    # Load data    
    numTraining = options.training
    numTest = options.test

    if(options.data=="faces"):
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
        rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
        rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
        rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)


    # Extract features
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    validationData = map(featureFunction, rawValidationData)
    testData = map(featureFunction, rawTestData)

    # Conduct training and testing
    print "Training..."
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print "Validating..."
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

    # do odds ratio computation if specified at command line
    if((options.odds) & (options.classifier == NB) ):
        label1, label2 = options.label1, options.label2
        features_odds = classifier.findHighOddsFeatures(label1,label2)
        if(options.classifier == NB):
            string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
        else:
            string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)        

        print string3
        printImage(features_odds)

    if((options.weights) & (options.classifier == PT)):
        for l in classifier.legalLabels:
            features_weights = classifier.findHighWeightFeatures(l)
            print ("=== Features with high weight for label %d ==="%l)
            printImage(features_weights)

if __name__ == '__main__':
    # Read input
    args, options = readCommand( sys.argv[1:] ) 
    # Run classifier
    runClassifier(args, options)
