from os import listdir
import os
import numpy as np
# import cv2
from scipy import ndimage
from multiprocessing import Pool
from keras.utils import to_categorical

from lib import util

PIXEL_SIZE = 1
classify = False
bin_class = False
small_obj_heights = False

def loadLocations(input_arr):
    locName = input_arr
    return {locName:Location.load(locName, 'all')}

class RawData(object):

    def __init__(self, locs):
        self.locs = locs

    @staticmethod
    def load(locNames='all', special_layers='all', new_data=None):
        print("in rawdata load")
        if locNames == 'all':
            print('1')
            locNames = listdir_nohidden('data/')
            print('1')
        if locNames == 'untrain':
            print('2')
            locNames = listdir_nohidden('data/_untrained/')
            print('2')
        if special_layers == 'all':
            print('3')
            if new_data is None:
#training
                print("four locations")
                locs = {}
                cores = 4
                chunksize = 1
                with Pool(processes=cores) as pool:
                    location_list_return = pool.map(loadLocations, locNames, chunksize)

                for i in location_list_return:
                    locs[list(i.keys())[0]] = i[list(i.keys())[0]]
#endtraining
            else:
                print('one testing location')
                locs = {n:Location.load(n, 'all') for n in locNames}

            print('3')
        else:
            # assumes dates is a dict, with keys being locNames and vals being special_layers
            print('4')
            locs = {n:Location.load(n, special_layers[n]) for n in locNames}
            print('4')
        return RawData(locs)

    def getClassificationOutput(self, locName, location):
        loc = self.locs[locName]
        return loc.obj_height_classification[location]

    def getOutput(self, locName, location):
        loc = self.locs[locName]
        return loc.layer_obj_heights[location]

    def getSpecialLayer(self, locName, special_layer):
        loc = self.locs[locName]
        # layer = loc.specialLayers[special_layer]
        return layer.layer_obj_heights

    def __repr__(self):
        return "Dataset({})".format(list(self.locs.values()))

class Location(object):

    def __init__(self, name, specialLayers, obj_heights=None, layers=None):
        self.name = name
        self.specialLayers = specialLayers
        self.layer_obj_heights = obj_heights if obj_heights is not None else self.loadLayerObjHeights()
        self.layers = layers if layers is not None else self.loadLayers()
        # what is the height and width of a layer of data
        self.layerSize = list(self.layers.values())[0].shape[:2]

        if classify:
            if bin_class:
                self.obj_height_classification = to_categorical(self.layer_obj_heights, 2)
            else:
                self.obj_height_classification = to_categorical(self.layer_obj_heights, 4)

    def loadLayers(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.name in untrainged_locNames:
            directory = cwd + '/data/_untrained/{}/'.format(self.name)
        else:
            directory = cwd + '/data/{}/'.format(self.name)

        dem = np.loadtxt(directory + 'dem.txt', delimiter=',') #util.openImg(folder+'dem.tif')
        slope = np.loadtxt(directory + 'slope.txt', delimiter=',')#util.openImg(folder+'slope.tif')
        band_1 = np.loadtxt(directory + 'band_1.txt', delimiter=',')#util.openImg(folder+'band_1.tif')
        band_2 = np.loadtxt(directory + 'band_2.txt', delimiter=',')#util.openImg(folder+'band_2.tif')
        band_3 = np.loadtxt(directory + 'band_3.txt', delimiter=',')#util.openImg(folder+'band_3.tif')
        band_4 = np.loadtxt(directory + 'band_4.txt', delimiter=',')#util.openImg(folder+'band_4.tif')
        ndvi = np.loadtxt(directory + 'ndvi.txt', delimiter=',')#util.openImg(folder+'ndvi.tif')
        aspect = np.loadtxt(directory + 'aspect.txt', delimiter=',')#util.openImg(folder+'aspect.tif')
        footprints = self.loadVeg(self.name)

        f_32 = [dem, slope, ndvi, aspect]
        # above_zero = [dem, slope]
        u_8 = [band_1, band_2, band_3, band_4]

        for l in f_32:
            l = l.astype('float32')


        for b in u_8:
            b = b.astype('uint8')
            b[b<0] = 0
            b[b>255] = 255

        grvi = np.divide(band_4, band_2, out=np.zeros_like(band_4), where=band_2!=0)

        layers = {'dem':dem,
                'slope':slope,
                'ndvi':ndvi,
                'aspect':aspect,
                'band_4':band_4,
                'band_3':band_3,
                'band_2':band_2,
                'band_1':band_1,
                'footprints': footprints,
                'grvi': grvi}

        for name, layer in layers.items():
            pass
        return layers

    # @staticmethod
    def loadLayerObjHeights(self):
        cwd = os.getcwd()
        untrained_locNames = listdir_nohidden('data/_untrained/')
        if self.name in untrained_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/obj_height.txt'.format(self.name)
        else:
            fname = cwd + '/data/{}/special_layers/obj_height.txt'.format(self.name)
        obj_heights = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        obj_heights = obj_heights.astype('float32')
        if classify:
            print('classify')
            if bin_class:
                obj_heights[obj_heights < 10] = 0
                obj_heights[obj_heights >= 10] = 1
            else:
                obj_heights[obj_heights < 5] = 0
                obj_heights[(obj_heights >= 5) & (obj_heights < 20)] = 1
                obj_heights[(obj_heights >= 20) & (obj_heights < 50)] = 2
                obj_heights[obj_heights >= 50] = 3

        if small_obj_heights:
            obj_heights[obj_heights<0] = 0
            obj_heights[obj_heights>150] = 150
            obj_heights = np.divide(obj_heights, 150, out=np.zeros_like(obj_heights))
        return obj_heights

    def loadVeg2(self):
        cwd = os.getcwd()
        untrained_locNames = listdir_nohidden('data/_untrained/')
        if self.name in untrained_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(self.name)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(self.name)

        veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_COLOR)
        veg = veg.astype('uint8')

        if veg is None:
            raise RuntimeError('Could not find veg for location {} for the layer'.format(locName))
        veg = np.zeros_like(veg)
        return veg

    @staticmethod
    def loadVeg(locName, specialLayers='veg'):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(locName)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(locName)

        veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_COLOR)
        veg = veg.astype('uint8')
        if veg is None:
            raise RuntimeError('Could not find veg for location {} for the layer'.format(locName))
        veg[veg!=0] = 255
        return veg

    @staticmethod
    def load(locName, specialLayers='all'):
        if specialLayers == 'all':
            special_layers = SpecialLayer.getVegLayer(locName)
        specialLayers = {layer_name:SpecialLayer(locName, layer_name) for layer_name in special_layers}

        return Location(locName, specialLayers)

    def __repr__(self):
        return "Location({}, {})".format(self.name, [d.layer_name for d in self.specialLayers.values()])

class SpecialLayer(object):

    def __init__(self, locName, layer_name, allVeg=None, footprints=None, obj_heights=None):
        self.locName = locName
        self.layer_name = layer_name
        self.allVeg = allVeg if allVeg is not None else self.loadAllVeg()
        self.footprints = footprints     if footprints   is not None else self.loadFootprints()
        self.obj_heights = obj_heights              if obj_heights is not None else self.loadObjHeights()

    def loadAllVeg(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(self.locName)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(self.locName)
        veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        veg = veg.astype('uint8')
        if veg is None:
            raise RuntimeError('Could not find veg for location {} for the layer {}'.format(self.locName, self.layer_name))
        veg[veg!=0] = 0
        return veg

    def loadFootprints(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/footprints.txt'.format(self.locName)
        else:
            fname = cwd + '/data/{}/special_layers/footprints.txt'.format(self.locName)

        not_veg = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        not_veg = not_veg.astype('uint8')
        if not_veg is None:
            raise RuntimeError('Could not open a footprint for the location {}'.format(self.locName))
        return not_veg


    def loadObjHeights(self):
        cwd = os.getcwd()
        untrainged_locNames = listdir_nohidden('data/_untrained/')
        if self.locName in untrainged_locNames:
            fname = cwd + '/data/_untrained/{}/special_layers/obj_height.txt'.format(self.locName)
        else:
            fname = cwd + '/data/{}/special_layers/obj_height.txt'.format(self.locName)
        obj_heights = np.loadtxt(fname, delimiter=',')#cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        obj_heights = obj_heights.astype('float32')
        if classify:
            print('classify')
            if bin_class:
                obj_heights[obj_heights < 10] = 0
                obj_heights[obj_heights >= 10] = 1
            else:
                obj_heights[obj_heights < 5] = 0
                obj_heights[(obj_heights >= 5) & (obj_heights < 20)] = 1
                obj_heights[(obj_heights >= 20) & (obj_heights < 50)] = 2
                obj_heights[obj_heights >= 50] = 3

        if small_obj_heights:
            obj_heights[obj_heights<0] = 0
            obj_heights[obj_heights>150] = 150
            obj_heights = np.divide(obj_heights, 150, out=np.zeros_like(obj_heights))

        return obj_heights


    def __repr__(self):
        return "specialLayer({},{})".format(self.locName, self.layer_name)


    @staticmethod
    def getVegLayer(locName):
        vegLayers = ['footprints']
        return vegLayers

def listdir_nohidden(path):
    '''List all the files in a path that are not hidden (begin with a .)'''
    result = []

    for f in listdir(path):
        if not f.startswith('.') and not f.startswith("_"):
            result.append(f)
    return result

if __name__ == '__main__':
    raw = RawData.load()
