from time import localtime, strftime
import os
import time
import tensorflow as tf
from keras import backend as K


print('importing keras...')
import keras.models
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.layers import BatchNormalization
from keras.optimizers import SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import random
print('done.')

classify = False
bin_class = False
GPU = False


try:
    from lib import preprocess
except:
    import preprocess



class BaseModel(object):

    DEFAULT_EPOCHS = 1
    DEFAULT_BATCHSIZE = 1000

    def __init__(self, kerasModel=None, preProcessor=None):
        self.kerasModel = kerasModel
        self.preProcessor = preProcessor

    def fit(self, trainingDataset, validatateDataset=None, epochs=DEFAULT_EPOCHS,batch_size=DEFAULT_BATCHSIZE):
        assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
        assert self.preProcessor is not None, "You must set the preProcessor within a subclass"

        print('training on ', trainingDataset)
        # get the actual samples from the collection of points
        (tinputs, toutputs), ptList = self.preProcessor.process(trainingDataset)
        if validatateDataset is not None:
            (vinputs, voutputs), ptList = self.preProcessor.process(validatateDataset)
            history = self.kerasModel.fit(tinputs, toutputs, batch_size=batch_size, epochs=epochs, validation_data=(vinputs, voutputs))
        else:
            history = self.kerasModel.fit(tinputs, toutputs, batch_size=batch_size, epochs=epochs)
        return history

    def predict(self, dataset):
        assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
        print("In predict")
        (inputs, outputs), ptList = self.preProcessor.process(dataset)
        results = self.kerasModel.predict(inputs).flatten()
        if mode:
            resultDict = {pt:random.utility(0.0, 1.0) for (pt, pred) in zip(ptList, results)}
        else:
            resultDict = {pt:pred for (pt, pred) in zip(ptList, results)}
        return resultDict

    def save(self, name=None):
        if name is None:
            name = strftime("%d%b%H_%M", localtime())
        if "models/" not in name:
            name = "models/" + name
        if not name.endswith('/'):
            name += '/'

        if not os.path.isdir(name):
            os.mkdir(name)

        className = str(self.__class__.__name__)
        with open(name+'class.txt', 'w') as f:
            f.write(className)
        self.kerasModel.save(name+'model.h5')

def load(modelFolder):
    if 'models/' not in modelFolder:
        modelFolder = 'models/' + modelFolder
    assert os.path.isdir(modelFolder), "{} is not a folder".format(modelFolder)

    if not modelFolder.endswith('/'):
        modelFolder += '/'

    modelFile = modelFolder + 'model.h5'
    model = keras.models.load_model(modelFile)

    objFile = modelFolder + 'class.txt'
    with open(objFile, 'r') as f:
        classString = f.read().strip()
    class_ = globals()[classString]
    obj = class_(kerasModel=model)

    return obj


class HeightModel(Sequential): #Model

    def __init__(self, preProcessor, weightsFileName=None):
        self.preProcessor = preProcessor

        kernelDiam = 2*self.preProcessor.AOIRadius+1

        super().__init__()
        # there is also the starting perim which is implicitly gonna be included
        nchannels = len(self.preProcessor.whichLayers)
        nchannels += 1
        input_shape = (kernelDiam, kernelDiam, nchannels)
        print(input_shape)


        self.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),
                        activation='relu', input_shape=input_shape)) #, input_shape=input_shape
        self.add(Conv2D(32, (3,3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Dropout(0.3))

        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Dropout(0.3))


        self.add(Flatten())
        self.add(Dense(96, kernel_initializer = 'normal', activation = 'relu',name='first_dense'))
        self.add(Dropout(0.3))
        self.add(Dense(160, kernel_initializer = 'normal', activation = 'relu',name='output'))

        if classify:
            if bin_class:
                print("IN BIN CLASS")
                self.add(Dense(2, activation='softmax'))

                self.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
            else:
                print("IN CLASS")
                self.add(Dense(4, activation='softmax'))

                self.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        else:
            print("IN REGRESS")
            self.add(Dense(1, kernel_initializer = 'normal', activation = 'relu',name='final_output'))

            self.compile(optimizer='adam',
                loss='mean_squared_error')

        if weightsFileName is not None:
            self.load_weights(weightsFileName)

    def fit(self, training, validate, pp, epochs=1):
        if GPU:
            trainPtList = training.toList(training.points)
            valPtList = validate.toList(validate.points)
            partition = {'train':[None] * len(trainPtList),
                        'validation':[None] * len(valPtList)}
            labels = {}
            t_idx = 0
            for pt in trainPtList:
                # locName, location = pt
                partition['train'][t_idx] = pt
                labels[pt] = t_idx
                t_idx += 1

            v_idx = 0
            for pt in valPtList:
                partition['validation'][v_idx] = pt
                labels[pt] = v_idx
                v_idx += 1

            aoi_size = (2 * pp.AOIRadius) + 1

            params = {'dim': (aoi_size, aoi_size),
              'batch_size': 5,
              'n_channels': len(pp.whichLayers) + 1,
              'shuffle': True,
              'whichLayers': pp.whichLayers,
              'AOIRadius': pp.AOIRadius,
              'dataset': training}

            # print("partition: ", partition)
            # print('labels: ', labels)


            training_generator = preprocess.DataGenerator(partition['train'], labels, **params)
            validation_generator = preprocess.DataGenerator(partition['validation'], labels, **params)


            history = super().fit_generator(generator=training_generator, epochs=epochs, validation_data=validation_generator, workers=0)
        else:
            print("JUST FIT")
            # get the actual samples from the collection of points
            (tinputs, toutputs), ptList = self.preProcessor.process(training)
            (vinputs, voutputs), ptList = self.preProcessor.process(validate)
            print('training on ', training)

            if classify:
                history = super().fit(tinputs, toutputs, batch_size=100000, epochs=epochs, validation_data=(vinputs, voutputs))
            else:
                history = super().fit(tinputs, toutputs, batch_size=100000, epochs=epochs, validation_data=(vinputs, voutputs))

        temp = self.saveWeights()
        return history

    def saveWeights(self, fname=None):
        if fname is None:
            timeString = time.strftime("%m%d-%H%M%S")
            fname = 'models/{}_'.format(timeString)
            if classify:
                if bin_class:
                    fname = fname + "classifyBIN"
                else:
                    fname = fname + "classify"
            else:
                fname = fname + "regress"

        self.save_weights(fname)
        return fname

    def predict(self, dataset, mode):
        print('start predict')
        (inputs, outputs), ptList = self.preProcessor.process(dataset)

        results = super().predict(inputs).flatten()

        if classify:
            if bin_class:
                big_results = []
                little_arr = []
                count = 0
                assert len(results)%2 == 0
                for i in results:
                    if count == 2:
                        count = 0
                        assert round(sum(little_arr), 3) == 1.0
                        big_results.append(little_arr)
                        little_arr = []

                    little_arr.append(i)
                    count = count + 1

                results = big_results
            else:
                big_results = []
                little_arr = []
                count = 0
                assert len(results)%4 == 0
                for i in results:
                    if count == 4:
                        count = 0
                        assert round(sum(little_arr), 3) == 1.0
                        big_results.append(little_arr)
                        little_arr = []

                    little_arr.append(i)
                    count = count + 1

                results = big_results

        if mode:
            resultDict = {pt:random.uniform(0.0,1.0) for (pt, pred) in zip(ptList, results)}
        else:
            resultDict = {pt:pred for (pt, pred) in zip(ptList, results)}
        print("end predict")
        return resultDict, results

class OurModel(BaseModel):

    def __init__(self, kerasModel=None):
        usedLayers = ['dem','ndvi', 'aspect', 'band_1', 'band_2', 'band_3', 'band_4', 'slope', 'grvi']
        AOIRadius = 5
        pp = preprocess.PreProcessor(usedLayers, AOIRadius)

        if kerasModel is None:
            kerasModel = self.createModel(pp)

        super().__init__(kerasModel, pp)

    @staticmethod
    def createModel(pp):
        # make our keras Model
        kernelDiam = 2*pp.AOIRadius+1
        ib = ImageBranch(len(pp.whichLayers), kernelDiam)

        kerasModel = ImageBranch(len(pp.whichLayers), kernelDiam)
        return kerasModel

class OurModel2(BaseModel):
    pass

if __name__ == '__main__':
    m = OurModel()
    m.save()

    n = load('models/15Nov09_41')
    print(n)
