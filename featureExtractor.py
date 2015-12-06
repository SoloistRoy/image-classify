# featureExtractor.py
# This file contains feature extraction methods

import samples
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70

def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether each pixel in the
    provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether each pixel in the
    provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = util.Counter()
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x,y)] = 1
            else:
                features[(x,y)] = 0
    return features

def enhancedFeatureExtractorDigit(datum):
    """
    Your feature extraction playground.

    You should return a util.Counter() of features for this datum (datum is of
    type samples.Datum).

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractorDigit(datum)

    "*** YOUR CODE HERE ***"

    return features

def enhancedFeatureExtractorFace(datum):
    """
    Your feature extraction playground for faces.
    """
    features = basicFeatureExtractorFace(datum)
    return features

