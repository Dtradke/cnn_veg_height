# preprocess.py
from collections import namedtuple
import numpy as np
import sys
import keras


try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from lib import util
except:
    import util

classify = False

class PreProcessor(object):
    '''What is responsible for extracting the used data from the dataset and then
    normalizing or doing any other steps before feeding it into the network.'''

    def __init__(self, whichLayers, AOIRadius):
        self.whichLayers = whichLayers
        self.AOIRadius = AOIRadius

    def process(self, dataset):
        '''Take a dataset and return the extracted inputs and outputs'''
        # create dictionaries mapping from Point to actual data from that Point

        aois = getSpatialData(dataset, self.whichLayers, self.AOIRadius)
        outs = getOutputs(dataset)


        # convert the dictionaries into lists, then arrays

        ptList = dataset.toList(dataset.points)

        i, o = [], []
        idx = 0
        for pt in ptList:
            locName, location = pt

            i.append(aois[locName, location])
            o.append(outs[locName, location])
        imgInputs = np.array(i)
        outputs = np.array(o)

        return (imgInputs, outputs), ptList


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, whichLayers, AOIRadius, dataset, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=7, shuffle=True):
        'Initialization'
        # print('IN INIT')
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.whichLayers = whichLayers
        self.AOIRadius = AOIRadius
        self.dataset = dataset

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # print('in on epoch end')
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        aois = getSpatialDataGPU(list_IDs_temp, self.whichLayers, self.AOIRadius, self.dataset) #list_IDs_temp is list of points
        outs = getOutputsGPU(self.dataset, list_IDs_temp)


        # Generate data, ID is point
        for i, ID in enumerate(list_IDs_temp):
            locName, location = ID

            X[i,] = aois[locName, location]
            y[i] = outs[locName, location]

        return X, y



def getSpatialDataGPU(ptsList, whichLayers, AOIRadius, dataset):
    # for each channel in the dataset, get all of the used data
    layers = {layerName:dataset.getAllLayers(layerName) for layerName in whichLayers}
    # now normalize them

    layers = normalizeLayers(layers)
    # now order them in the whichLayers order, stack them, and pad them
    paddedLayers = stackAndPad(layers, whichLayers, dataset, AOIRadius)
    # now extract out the aois around each point
    result = {}
    for pt in ptsList:
        locName, location = pt
        padded = paddedLayers[locName]
        aoi = extract(padded, location, AOIRadius)
        result[(locName, location)] = aoi
    return result

def getSpatialData(dataset, whichLayers, AOIRadius):
    # for each channel in the dataset, get all of the used data
    layers = {layerName:dataset.getAllLayers(layerName) for layerName in whichLayers}
    # now normalize them
    layers = normalizeLayers(layers)
    # now order them in the whichLayers order, stack them, and pad them
    paddedLayers = stackAndPad(layers, whichLayers, dataset, AOIRadius)
    # now extract out the aois around each point
    result = {}
    for pt in dataset.toList(dataset.points):
        locName, location = pt
        padded = paddedLayers[locName]
        aoi = extract(padded, location, AOIRadius)
        result[(locName, location)] = aoi
    return result

def normalizeLayers(layers):
    result = {}
    for name, data in layers.items():
        result[name] = normalizeNonElevations(data)
    return result

def normalizeElevations(dems):
    avgElevation = {}
    validIndicesDict = {}
    ranges = {}
    for locName, dem in dems.items():
        validIndices = np.where(np.isfinite(dem))
        validIndicesDict[locName] = validIndices
        validPixels = dem[validIndices]
        avgElevation[locName] = np.mean(validPixels)
        ranges[locName] = validPixels.max()-validPixels.min()

    maxRange = max(ranges.values())
    results = {}
    for locName, dem in dems.items():
        validIndices = validIndicesDict[locName]
        validPixels = dem[validIndices]
        normed = util.normalize(validPixels)
        blank = np.zeros_like(dem, dtype=np.float32)
        thisRange = ranges[locName]
        scaleFactor = thisRange/maxRange
        blank[validIndices] = scaleFactor * normed
        results[locName] = blank
    return results

def normalizeNonElevations(nonDems):
    splitIndices = [0]
    validPixelsList = []
    validIndicesList = []
    names = list(nonDems.keys())
    for name in names:
        layer = nonDems[name]
        validIndices = np.where(np.isfinite(layer))
        validPixels = layer[validIndices]

        validPixelsList += validPixels.tolist()
        splitIndices.append(splitIndices[-1] + len(validPixels))
        validIndicesList.append(validIndices)

    arr = np.array(validPixelsList)
    normed = util.normalize(arr)
    splitIndices = splitIndices[1:]
    splitBackUp = np.split(normed, splitIndices)
    results = {}
    for name, validIndices, normedPixels in zip(names,validIndicesList,splitBackUp):
        src = nonDems[name]
        img = np.zeros_like(src, dtype=np.float32)
        img[validIndices] = normedPixels
        results[name] = img
    return results

def getOutputsGPU(dataset, ptsList):
    result = {}
    for pt in ptsList:
        locName, location = pt #location is tuples of (y, x) locations
        if classify:
            out = dataset.data.getClassificationOutput(locName, location)
        else:
            out = dataset.data.getOutput(locName, location)

        result[(locName, location)] = out
    return result


def getOutputs(dataset):
    result = {}
    for pt in dataset.toList(dataset.points):
        locName, location = pt #location is tuples of (y, x) locations
        if classify:
            out = dataset.data.getClassificationOutput(locName, location)
        else:
            out = dataset.data.getOutput(locName, location)

        result[(locName, location)] = out

    return result

def stackAndPad(layerDict, whichLayers, dataset, AOIRadius):
    result = {}
    for locName in dataset.getUsedLocNames():
        sp = dataset.data.locs[locName].loadVeg(locName)
        # sp = foots
        sp[sp!=0]=1

        layers = [layerDict[layerName][locName] for layerName in whichLayers]
        layers = [sp] + layers
        stacked = np.dstack(layers)
        r = AOIRadius
        # pad with zeros around border of image
        padded = np.lib.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
        result[locName] = padded
    return result

def extract(padded, location, AOIRadius):
    '''Assume padded is bordered by radius self.inputSettings.AOIRadius'''
    y,x = location
    r = AOIRadius
    lox = r+(x-r)
    hix = r+(x+r+1)
    loy = r+(y-r)
    hiy = r+(y+r+1)
    aoi = padded[loy:hiy,lox:hix]
    return aoi



# =========================================================

if __name__ == '__main__':
    import rawdata
    import dataset
    data = rawdata.RawData.load()
    ds = dataset.Dataset(data, points=dataset.Dataset.vulnerablePixels)
