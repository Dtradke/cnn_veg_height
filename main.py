import numpy as np
import random
import sys
import time
import os


from lib import rawdata
from lib import dataset
from lib import metrics
from lib import viz
from lib import preprocess
from lib import util
from multiprocessing import Pool

SAMPLE_SIZE = 2000

classify = False
bin_class = False
rand = False

def makeNewAreaDataset():
    new_data = rawdata.RawData.load(locNames='untrain', special_layers='all', new_data='not_none')
    newDataSet = dataset.Dataset(new_data, dataset.Dataset.vulnerablePixels)
    pointLst = newDataSet.toListTest(newDataSet.points)
    # ptList = masterDataSet.sample(sampleEvenly=False)
    # pointLst = random.sample(pointLst, SAMPLE_SIZE)
    test = dataset.Dataset(new_data, pointLst)
    return test

def makeSmallDatasets(pass_arr):
    data = pass_arr[0]
    set_type = pass_arr[1]
    type_str = pass_arr[2]
    return_dict = {}
    return_dict[type_str] = dataset.Dataset(data, set_type)
    return return_dict


def openDatasets():
    data = rawdata.RawData.load(locNames='all', special_layers='all')
    masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels) #this loops through vulnerablePixels for each location... should grab all veg image
    sample_size = 100
    print("SAMPLE SIZE: ", sample_size)
    ptList = masterDataSet.sample(goalNumber=sample_size, sampleEvenly=False) #goalNumber=sample_size,
    trainPts, validatePts, testPts =  util.partition(ptList, ratios=[.7, .8])#.85,.99
    train = dataset.Dataset(data, trainPts)
    validate = dataset.Dataset(data, validatePts)
    test = dataset.Dataset(data, testPts)

    return train, validate, test

def openAndPredict(weightsFile):
    from lib import model

    test = makeNewAreaDataset()
    test.save('testOtherLocation')
    mod = getModel(weightsFile)
    predictions, _ = mod.predict(test, rand)
    util.savePredictions(predictions)
    res = viz.visualizePredictions(test, predictions)
    viz.showPredictions(res)
    return test, predictions

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
            for i in range(wanted_parts) ]


def openAndTrain(train_dataset=None, val_dataset=None, test_dataset=None):
    start_time = time.time()
    from lib import model
    if train_dataset is None:
        train, validate, test = openDatasets()
        train_amt = 0
        val_amt = 0
        test_amt = 0
        for i in list(train.points.keys()):
            train_amt = train_amt + len(train.points[i])

        for i in list(validate.points.keys()):
            val_amt = val_amt + len(validate.points[i])

        for i in list(test.points.keys()):
            test_amt = test_amt + len(test.points[i])

        train_fname = train.save('train_' + str(train_amt) + '_')
        test_fname = test.save('test_' + str(test_amt) + '_')
        validate_fname = validate.save('validate_' + str(val_amt) + '_')
        dataset_fname = 'train_' + train_fname + "--val_" + validate_fname# + "--test_" + test_fname
    else:
        datasetfname = 'output/datasets/'
        train = dataset.openDataset(datasetfname + train_dataset)
        validate = dataset.openDataset(datasetfname + val_dataset)
        test = dataset.openDataset(datasetfname + test_dataset)
        test_fname = test_dataset
        dataset_fname = 'train_' + train_dataset + "--val_" + val_dataset# + "--test_" + test_dataset

    epochs = 30

    mod, pp = getModel()
    mod.fit(train, validate, pp, epochs)
    mod_string = mod.saveWeights()
    print("Model: ", mod_string[7:])

    #on testing locations
    dataset_fname = dataset_fname + '_on_' + test_fname
    predictions, _ = mod.predict(test, rand)
    util.savePredictions(predictions, mod_string[7:])
    example2(predictions, test, mod_string, dataset_fname)


    try:
        if classify:
            if bin_class:
                cwd = os.getcwd()
                path = cwd + '/output/datasets/Test-Site0401-140603classBIN_' #on SyN cluster
                real_test = dataset.openDataset(path, 'untrain')
                real_test_fname = 'Test-Site0401-140603classBIN_'
            else:
                cwd = os.getcwd()
                path = cwd + '/output/datasets/Test-Site0401-140603class_' #on SyN cluster
                real_test = dataset.openDataset(path, 'untrain')
                real_test_fname = 'Test-Site0401-140603class_'
        else:
            cwd = os.getcwd()
            path = cwd + '/output/datasets/Test-Site0401-095013regress_' #on SyN cluster
            real_test = dataset.openDataset(path, 'untrain')
            real_test_fname = 'Test-Site0401-095013regress_'
    except:
        real_test = makeNewAreaDataset()
        real_test_fname = real_test.save("Test-Site")
        # real_test_fname = dataset.Dataset.save(real_test)

    dataset_fname = dataset_fname + '_on_' + real_test_fname
    predictions, _ = mod.predict(real_test, rand)
    util.savePredictions(predictions, mod_string[7:])
    example2(predictions, real_test, mod_string, dataset_fname)

    total_time = time.time() - start_time
    print("total minutes: ", total_time/60)
    print("total hours: ", total_time/3600)
    print("model: ", mod_string[7:])

    showModel(mod_string[7:])

    return real_test, predictions

# def example2(mod, dataset):
def example2(predictions, dataset, mod, training_data):
    print("evaluating")
    print("SIZE OF PREDICTIONS: " , len(predictions))
    results = viz.evaluate_heights(predictions, dataset)
    results['dataset'] = training_data
    results['model'] = mod
    util.saveEvaluation(results, training_data)

def reloadPredictions():
    predFName = "09Nov10:39.csv"
    predictions = util.openPredictions('output/predictions/'+predFName)
    test = dataset.Dataset.open("output/datasets/test.json")
    return test, predictions

def getModel(weightsFile=None):
    from lib import model
    print('in getModel')
    usedLayers = ['dem','ndvi', 'aspect', 'slope', 'footprints', 'band_1', 'band_2', 'band_3', 'band_4', 'grvi'] #, 'evi'
    AOIRadius = 11
    pp = preprocess.PreProcessor(usedLayers, AOIRadius)

    mod = model.HeightModel(pp, weightsFile)
    return mod, pp

def getRealHeights(dataset):
    predictions = {}
    obj_height_list = []

    ptList = dataset.points[list(dataset.points.keys())[0]]

    count = 0

    for pt in ptList:
        locName, location = pt

        if count == 0 or count == 1 and classify:
            if count == 0:
                obj_height_list.append(np.argmax(0))
            else:
                obj_height_list.append(np.argmax(3))
            count = count + 1
        else:
            loc = dataset.data.locs[locName]
            if classify or bin_class:
                obj_height_list.append(np.argmax(loc.obj_height_classification[location]) + 1)
            else:
                obj_height_list.append(loc.layer_obj_heights[location])


    resultDict = {pt:pred for (pt, pred) in zip(ptList, obj_height_list)}
    return resultDict


def example():
    if sys.argv[1] == 'all':
        modfname = 'all'
        # file_loc = 'untrain'
    else:
        modfname = 'models/syn_models/' + sys.argv[1] #syn_models/

    file_loc = 'untrain'

    datasetfname = 'output/datasets/' + sys.argv[2]
    print("working")

    test = dataset.openDataset(datasetfname, file_loc)

    if modfname != 'all':
        mod, _ = getModel(modfname)
        predictions, _ = mod.predict(test, rand)
    else:
        predictions = getRealHeights(test)
        print('got ground truth...')
        mod = 'all'

    example2(predictions, test, mod, datasetfname)

    preResu = [ None ] * len(predictions)

    print("SIZE OF PREDICTIONS: " , len(predictions))

    idx = 0
    for pre in predictions:
        preResu[idx] = predictions.get(pre)
        idx = idx + 1

    if rand:
        print("THIS IS A RANDOM TEST!!!")

    # for pt, pred in predictions.items():


    res = viz.visualizePredictions(test, predictions, preResu)
    viz.showPredictions(res)


def showModel(modelStr):

    modfname = 'models/' + str(modelStr)
    mod, _ = getModel(modfname)
    viz.saveModel(mod, modelStr)

if len(sys.argv) == 4:
    print('Training a new model with old datasets...')
    # python3 main.py train0325-220335_ validate0325-220529_ test0325-220528_ | tee output_aqua.out&
    openAndTrain(sys.argv[1], sys.argv[2], sys.argv[3])
elif len(sys.argv) == 3:
    print('Making test picture with model...')
    # command: python3 main.py all Test_Site_0320-170405_
    example()
elif len(sys.argv) == 2:
    # command: python3 main.py model_name
    print('Showing model...')
    showModel(sys.argv[1])
elif len(sys.argv) == 1:
    # command: python3 main.py
    print('Training a new model...')
    openAndTrain()
else:
    # command: python3 main.py make a new dataset now
    print('making new dataset')
    test = makeNewAreaDataset()
    dataset.Dataset.save(test)
