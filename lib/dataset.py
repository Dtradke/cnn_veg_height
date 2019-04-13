import math
from collections import namedtuple
import random
import json
import time
from time import localtime, strftime
from multiprocessing import Pool


import numpy as np
# import cv2
from lib import rawdata
from lib import viz
from keras.preprocessing.image import ImageDataGenerator

AOIRadius = 11

classify = False
bin_class = False
small_obj_heights = False


class Dataset(object):
    '''A set of Point objects'''
    VULNERABLE_RADIUS = 500

    def __init__(self, data, points='all'):
        print('creating new dataset with points type: ', type(points))
        self.data = data

        self.points = points
        if points=='all':
            points = Dataset.allPixels
        if hasattr(points, '__call__'):
            # points is a filter function
            filterFunc = points
            self.points = self.filterPoints(self.data, filterFunc)


        if type(points) == list:
            print(len(points))
            list_split = self.split_list(points, 40)
            total_dataset_dict = {}

            cores = 40
            chunksize = 1
            with Pool(processes=cores) as pool:
                dataset_arr = pool.map(self.toDict, list_split, chunksize)

            for i in dataset_arr:
                for j in i.keys():
                    total_dataset_dict[j] = []

            for i in dataset_arr:
                for j in i.keys():
                    total_dataset_dict[j] = total_dataset_dict[j] + i[j]

            self.points = total_dataset_dict #self.toDict(points)


        the_type = type(self.points)
        comp = type({})
        assert the_type == comp

    @staticmethod
    def split_list(alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
                for i in range(wanted_parts) ]

    def getUsedLocNames(self):
        results = []
        locNames = self.points.keys()
        for loc in locNames:
            results.append(loc)
        return results

    def getAllLayers(self, layerName):
        result = {}
        allLocNames = list(self.points.keys())
        for locName in allLocNames:
            loc = self.data.locs[locName]
            layer = loc.layers[layerName]
            result[locName] = layer
        return result

    def __len__(self):
        total = 0
        for locName, locDict in self.points.items():
            for ptList in locDict.values():
                total += len(ptList)
        return total

    def save(self, fname=None):
        if classify:
            if bin_class:
                timeString = time.strftime("%m%d-%H%M%S" + "classBIN")
            else:
                timeString = time.strftime("%m%d-%H%M%S" + "class")
        else:
            timeString = time.strftime("%m%d-%H%M%S" + "regress")
        if fname is None:
            fname = timeString
        else:
            fname = fname + timeString
        if not fname.startswith("output/datasets/"):
            fname = "output/datasets/" + fname
        if not fname.endswith('.json'):
            fname = fname + '_'

        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(MyEncoder, self).default(obj)

        with open(fname, 'w') as fp:
            json.dump(self.points, fp, cls=MyEncoder, sort_keys=True, indent=4)

        return fname

    @staticmethod
    def toList(pointDict):
        '''Flatten the point dictionary to a list of Points'''
        result = []
        for locName in pointDict:
            points = pointDict[locName]
            result.extend(points)
        return result


    @staticmethod
    def toListTest(pointDict):
        '''Flatten the point dictionary to a list of Points'''
        result = []
        for locName in pointDict:
            points = pointDict[locName].values()
            for p in points:
                result.extend(p)

        return result

    @staticmethod
    def toDict(pointList):
        locs = {}
        locsCount = {}
        locsFillCount = {}
        for p in pointList:
            locName, _ = p
            if locName not in locsCount:
                locsCount[locName] = 1
            else:
                current = locsCount[locName]
                current = current + 1
                locsCount[locName] = current

        for name in locsCount.keys():
            locs[name] = [None] * locsCount[name]
            locsFillCount[name] = 0

        for p in range(len(pointList)):
            locName, location = pointList[p]

            if pointList[p] not in locs[locName]:
                locs[locName][locsFillCount[locName]] = pointList[p]
                locsFillCount[locName] = locsFillCount[locName] + 1

        return locs

    @staticmethod
    def filterPoints(data, filterFunction):
        '''Return all the points which satisfy some filterFunction'''
        points = {}
        locations = data.locs.values()
        for loc in locations:
            dictOfLoc = {}
            points[loc.name] = dictOfLoc
            oh = loc.loadVeg(loc.name, 'all')

            y_x = filterFunction(loc, loc)
            dictOfLoc[loc] = [Point(loc.name,l) for l in y_x]
        return points


    def sample(self, goalNumber='max', sampleEvenly=True):
        assert goalNumber == 'max' or (type(goalNumber)==int and goalNumber%2==0)
        height_res = self.makeDay2HighLowMap()

        max_pixel_amt = [0] * 10

        if classify:
            if bin_class:
                [print(loc, " : ", len(class1), len(class2)) for loc, (class1, class2) in height_res.items()]
                limits = {loc:min(len(class1), len(class2)) for loc, (class1, class2) in height_res.items()}
            else:
                [print(loc, " : ", len(class1), len(class2), len(class3), len(class4)) for loc, (class1, class2, class3, class4) in height_res.items()]
                limits = {loc:min(len(class1), len(class2), len(class3), len(class4)) for loc, (class1, class2, class3, class4) in height_res.items()}
        else:
            [print(loc, " : ", len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus)) for loc, (underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus) in height_res.items()]
            # find the limiting size for each location
            limits = {loc:min(len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus)) for loc, (underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus) in height_res.items()}

        print(limits)

        if sampleEvenly:
            # we must get the same number of samples from each day
            # don't allow a large location to have a bigger impact on training
            if goalNumber == 'max':
                # get as many samples as possible while maintaining even sampling
                samplesPerLoc = min(limits.values())
                print("samplesPerLoc", samplesPerLoc)
            else:
                # aim for a specific number of samples and sample evenly
                maxSamples = (2 * min(limits.values())) * len(limits)
                if goalNumber > maxSamples:
                    raise ValueError("Not able to get {} samples while maintaining even sampling from the available {}.".format(goalNumber, maxSamples))
                nlocs = len(limits)
                samplesPerLoc = goalNumber/(2*nlocs)
                samplesPerLoc = int(math.ceil(samplesPerLoc))
        else:
            # we don't care about sampling evenly. Larger Days will get more samples
            if goalNumber == 'max':
                # get as many samples as possible, whatever it takes
                samplesPerLoc = 'max'
            else:
                # aim for a specific number of samples and don't enforce even sampling
                samplesPerLoc = goalNumber/10
                maxSamples = sum(limits.values()) * 10
                if goalNumber > maxSamples:
                    raise ValueError("Not able to get {} samples from the available {}.".format(goalNumber, maxSamples))
        # order the days from most limiting to least limiting
        locs = sorted(limits, key=limits.get)
        tallSamples = []
        shortSamples = []
        underTwoSamples = []
        twoFiveSamples = []
        fiveTenSamples = []
        tenTwentySamples = []
        twentyThirtySamples = []
        thirtyFourtySamples = []
        fourtyFiftySamples = []
        fifty75Samples = []
        seven5HundSamples = []
        hundPlusSamples = []

        class1Samples = []
        class2Samples = []
        class3Samples = []
        class4Samples = []

        if classify:
            if bin_class:
                class1, class2 = [], []

                for i, loc in enumerate(locs):
                    class1_loc, class2_loc = height_res[loc] #tall, short

                    class1.extend(class1_loc)
                    class2.extend(class2_loc)

                random.shuffle(class1)
                random.shuffle(class2)

                if sampleEvenly:
                    print('now samplesPerLoc', samplesPerLoc)
                    class1Samples.extend(class1[:samplesPerLoc])
                    class2Samples.extend(class2[:samplesPerLoc])
                else:
                    if samplesPerLoc == 'max':
                        nsamples = min(len(class1), len(class2))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])
                    else:
                        samplesToGo = goalNumber/2 - len(class1Samples)
                        locsToGo = len(locs)-i
                        goalSamplesPerLoc = int(math.ceil(samplesToGo/locsToGo))
                        nsamples = min(goalSamplesPerLoc,len(class1), len(class2))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])

                # now shuffle, trim and split the samples
                print('length of all samples', len(class1), len(class2))
                random.shuffle(class1Samples)
                random.shuffle(class2Samples)

                if goalNumber != 'max':
                    class1Samples = class1Samples[:goalNumber//2]
                    class2Samples = class2Samples[:goalNumber//2]

                samples = class1Samples + class2Samples
            else:
                class1, class2, class3, class4 = [], [], [], []

                for i, loc in enumerate(locs):
                    class1_loc, class2_loc, class3_loc, class4_loc = height_res[loc] #tall, short

                    class1.extend(class1_loc)
                    class2.extend(class2_loc)
                    class3.extend(class3_loc)
                    class4.extend(class4_loc)

                random.shuffle(class1)
                random.shuffle(class2)
                random.shuffle(class3)
                random.shuffle(class4)

                if sampleEvenly:
                    print('now samplesPerLoc', samplesPerLoc)
                    class1Samples.extend(class1[:samplesPerLoc])
                    class2Samples.extend(class2[:samplesPerLoc])
                    class3Samples.extend(class3[:samplesPerLoc])
                    class4Samples.extend(class4[:samplesPerLoc])
                else:
                    if samplesPerLoc == 'max':
                        nsamples = min(len(class1), len(class2), len(class3), len(class4))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])
                        class3Samples.extend(class3[:nsamples])
                        class4Samples.extend(class4[:nsamples])
                    else:
                        samplesToGo = goalNumber/2 - len(class1Samples)
                        locsToGo = len(locs)-i
                        goalSamplesPerLoc = int(math.ceil(samplesToGo/locsToGo))
                        nsamples = min(goalSamplesPerLoc,len(class1), len(class2), len(class3), len(class4))
                        class1Samples.extend(class1[:nsamples])
                        class2Samples.extend(class2[:nsamples])
                        class3Samples.extend(class3[:nsamples])
                        class4Samples.extend(class4[:nsamples])

                # now shuffle, trim and split the samples
                print('length of all samples', len(class1), len(class2), len(class3), len(class4))
                # random.shuffle(tallSamples)
                # random.shuffle(shortSamples)
                random.shuffle(class1Samples)
                random.shuffle(class2Samples)
                random.shuffle(class3Samples)
                random.shuffle(class4Samples)

                if goalNumber != 'max':
                    class1Samples = class1Samples[:goalNumber//4]
                    class2Samples = class2Samples[:goalNumber//4]
                    class3Samples = class3Samples[:goalNumber//4]
                    class4Samples = class4Samples[:goalNumber//4]

                samples = class1Samples + class2Samples + class3Samples + class4Samples

        else:
            underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus = [], [], [], [], [], [], [], [], [], []

            for i, loc in enumerate(locs):
                underTwo_loc, twoFive_loc, fiveTen_loc, tenTwenty_loc, twentyThirty_loc, thirtyFourty_loc, fourtyFifty_loc, fifty75_loc, seven5Hund_loc, hundPlus_loc = height_res[loc] #tall, short

                underTwo.extend(underTwo_loc)
                twoFive.extend(twoFive_loc)
                fiveTen.extend(fiveTen_loc)
                tenTwenty.extend(tenTwenty_loc)
                twentyThirty.extend(twentyThirty_loc)
                thirtyFourty.extend(thirtyFourty_loc)
                fourtyFifty.extend(fourtyFifty_loc)
                fifty75.extend(fifty75_loc)
                seven5Hund.extend(seven5Hund_loc)
                hundPlus.extend(hundPlus_loc)

            random.shuffle(underTwo)
            random.shuffle(twoFive)
            random.shuffle(fiveTen)
            random.shuffle(tenTwenty)
            random.shuffle(twentyThirty)
            random.shuffle(thirtyFourty)
            random.shuffle(fourtyFifty)
            random.shuffle(fifty75)
            random.shuffle(seven5Hund)
            random.shuffle(hundPlus)

            if sampleEvenly:
                print('now samplesPerLoc', samplesPerLoc)
                underTwoSamples.extend(underTwo[:samplesPerLoc])
                twoFiveSamples.extend(twoFive[:samplesPerLoc])
                fiveTenSamples.extend(fiveTen[:samplesPerLoc])
                tenTwentySamples.extend(tenTwenty[:samplesPerLoc])
                twentyThirtySamples.extend(twentyThirty[:samplesPerLoc])
                thirtyFourtySamples.extend(thirtyFourty[:samplesPerLoc])
                fourtyFiftySamples.extend(fourtyFifty[:samplesPerLoc])
                fifty75fifty75Samples.extend(fifty75[:samplesPerLoc])
                seven5HundSamples.extend(seven5Hund[:samplesPerLoc])
                hundPlusSamples.extend(hundPlus[:samplesPerLoc])
            else:
                if samplesPerLoc == 'max':
                    nsamples = min(len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus))
                    underTwoSamples.extend(underTwo[:nsamples])
                    twoFiveSamples.extend(twoFive[:nsamples])
                    fiveTenSamples.extend(fiveTen[:nsamples])
                    tenTwentySamples.extend(tenTwenty[:nsamples])
                    twentyThirtySamples.extend(twentyThirty[:nsamples])
                    thirtyFourtySamples.extend(thirtyFourty[:nsamples])
                    fourtyFiftySamples.extend(fourtyFifty[:nsamples])
                    fifty75Samples.extend(fifty75[:nsamples])
                    seven5HundSamples.extend(seven5Hund[:nsamples])
                    hundPlusSamples.extend(hundPlus[:nsamples])
                else:
                    samplesToGo = goalNumber/2 - len(underTwoSamples)
                    locsToGo = len(locs)-i
                    goalSamplesPerLoc = int(math.ceil(samplesToGo/locsToGo))
                    nsamples = min(goalSamplesPerLoc,len(underTwo), len(twoFive), len(fiveTen), len(tenTwenty), len(twentyThirty), len(thirtyFourty), len(fourtyFifty), len(fifty75), len(seven5Hund), len(hundPlus))
                    underTwoSamples.extend(underTwo[:nsamples])
                    twoFiveSamples.extend(twoFive[:nsamples])
                    fiveTenSamples.extend(fiveTen[:nsamples])
                    tenTwentySamples.extend(tenTwenty[:nsamples])
                    twentyThirtySamples.extend(twentyThirty[:nsamples])
                    thirtyFourtySamples.extend(thirtyFourty[:nsamples])
                    fourtyFiftySamples.extend(fourtyFifty[:nsamples])
                    fifty75Samples.extend(fifty75[:nsamples])
                    seven5HundSamples.extend(seven5Hund[:nsamples])
                    hundPlusSamples.extend(hundPlus[:nsamples])

            # now shuffle, trim and split the samples
            print('length of all samples', len(underTwoSamples), len(twoFiveSamples), len(fiveTenSamples), len(tenTwentySamples), len(twentyThirtySamples), len(thirtyFourtySamples), len(fourtyFiftySamples), len(fifty75Samples), len(seven5HundSamples), len(hundPlusSamples))
            random.shuffle(underTwoSamples)
            random.shuffle(twoFiveSamples)
            random.shuffle(fiveTenSamples)
            random.shuffle(tenTwentySamples)
            random.shuffle(twentyThirtySamples)
            random.shuffle(thirtyFourtySamples)
            random.shuffle(fourtyFiftySamples)
            random.shuffle(fifty75Samples)
            random.shuffle(seven5HundSamples)
            random.shuffle(hundPlusSamples)
            if goalNumber != 'max':
                underTwoSamples = underTwoSamples[:goalNumber//10]
                twoFiveSamples = twoFiveSamples[:goalNumber//10]
                fiveTenSamples = fiveTenSamples[:goalNumber//10]
                tenTwentySamples = tenTwentySamples[:goalNumber//10]
                twentyThirtySamples = twentyThirtySamples[:goalNumber//10]
                thirtyFourtySamples = thirtyFourtySamples[:goalNumber//10]
                fourtyFiftySamples = fourtyFiftySamples[:goalNumber//10]
                fifty75Samples = fifty75Samples[:goalNumber//10]
                seven5HundSamples = seven5HundSamples[:goalNumber//10]
                hundPlusSamples = hundPlusSamples[:goalNumber//10]
            samples = underTwoSamples + twoFiveSamples + fiveTenSamples + tenTwentySamples + twentyThirtySamples + thirtyFourtySamples + fourtyFiftySamples + fifty75Samples + seven5HundSamples + hundPlusSamples

        random.shuffle(samples)
        print(len(samples), sum(limits.values()))

        return samples

    def makeDay2HighLowMap(self):
        result = {}
        for locName, vegDict in self.points.items():
            for layer, ptList in vegDict.items():
                if classify:
                    if bin_class:
                        heights = layer.obj_height_classification
                        class1, class2 = [], []
                        for pt in ptList:
                            _,location = pt

                            if heights[location][0] == 1:
                                class1.append(pt)
                            elif heights[location][1] == 1:
                                class2.append(pt)

                        result[(locName, 'allVeg')] = (class1, class2)
                    else:
                        heights = layer.obj_height_classification
                        class1, class2, class3, class4 = [], [], [], []
                        for pt in ptList:
                            _,location = pt

                            if heights[location][0] == 1:
                                class1.append(pt)
                            elif heights[location][1] == 1:
                                class2.append(pt)
                            elif heights[location][2] == 1:
                                class3.append(pt)
                            elif heights[location][3] == 1:
                                class4.append(pt)

                        result[(locName, 'allVeg')] = (class1, class2, class3, class4)
                else:
                    heights = layer.layer_obj_heights
                    tall, short = [], []

                    if small_obj_heights:
                        two = 2/150
                        five = 5/150
                        ten = 10/150
                        twenty = 20/150
                        thirty = 30/150
                        fourty = 40/150
                        fifty = 50/150
                        seven_five = 75/150
                        hund = 100/150
                    else:
                        two = 2
                        five = 5
                        ten = 10
                        twenty = 20
                        thirty = 30
                        fourty = 40
                        fifty = 50
                        seven_five = 75
                        hund = 100

                    underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus = [], [], [], [], [], [], [], [], [], []
                    for pt in ptList:
                        _ ,location = pt

                        if heights[location] < two:
                            underTwo.append(pt)
                        elif heights[location] < five:
                            twoFive.append(pt)
                        elif heights[location] < ten:
                            fiveTen.append(pt)
                        elif heights[location] < twenty:
                            tenTwenty.append(pt)
                        elif heights[location] < thirty:
                            twentyThirty.append(pt)
                        elif heights[location] < fourty:
                            thirtyFourty.append(pt)
                        elif heights[location] < fifty:
                            fourtyFifty.append(pt)
                        elif heights[location] < seven_five:
                            fifty75.append(pt)
                        elif heights[location] < hund:
                            seven5Hund.append(pt)
                        else:
                            hundPlus.append(pt)


                    result[(locName, 'allVeg')] = (underTwo, twoFive, fiveTen, tenTwenty, twentyThirty, thirtyFourty, fourtyFifty, fifty75, seven5Hund, hundPlus) #(tall, short)
        return result

    @staticmethod
    def allPixels(loc):
        return list(np.ndindex(loc.layerSize))

    @staticmethod
    def vulnerablePixels(loc, location, radius=VULNERABLE_RADIUS):
        '''Return the indices of the pixels that are vegetation'''
        startingVeg = location.loadVeg(location.name, 'all')

        neg_aoi = 0 - AOIRadius
        #only grabs pixels within the AOIRadius to not sample off of the edge
        startingVeg = startingVeg[AOIRadius: neg_aoi, AOIRadius: neg_aoi]
        ys, xs = np.where(startingVeg == 0) #border
        ys = ys + AOIRadius
        xs = xs + AOIRadius
        return list(zip(ys, xs))

    def __repr__(self):
        # shorten the string repr of self.points
        return "Dataset({}, with {} points)".format(self.data, len(self.toList(self.points)))

# create a class that represents a spatial and temporal location that a sample lives at
Point = namedtuple('Point', ['LocName', 'location'])

def split_list2(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
            for i in range(wanted_parts) ]


def make_newPtList(pts_arr):
    newPtList = [ None ] * len(pts_arr)
    idx = 0

    for name, loc in pts_arr:
        newPtList[idx] = Point(name, tuple(loc))
        idx = idx + 1

    return newPtList


def openDataset(fname, file_loc=None):
    print('in openDataset')
    with open(fname, 'r') as fp:
        print('in openDataset with')
        if file_loc == 'untrain':
            print('UNTRAIN')
            data = rawdata.RawData.load(locNames='untrain', special_layers='all')
        else:
            print('NOT UNTRAIN')
            data = rawdata.RawData.load(locNames='all', special_layers='all')
        print('still in with')
        pts = json.load(fp)
        newLocDict =  {}

        for locName, pts_arr in pts.items():
            print('in dataset for loop')
            print("in inner dataset for loop")
            newPtList = []


            list_split = split_list2(pts_arr, 2)


            cores = 2
            chunksize = 1
            with Pool(processes=cores) as pool:
                newPtList_arr = pool.map(make_newPtList, list_split, chunksize)


            for i in newPtList_arr:
                newPtList.extend(i)

            newLocDict[locName] = newPtList
            print("SIZE OF PTLIST: " , len(newPtList))


        return Dataset(data, newLocDict)

if __name__ == '__main__':
    d = rawdata.RawData.load()
