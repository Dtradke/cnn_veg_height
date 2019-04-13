from time import localtime, strftime
import csv


import numpy as np
import cv2
import random
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    print('Successfully imported pyplot')
except:
    print('Failed to import pyplot ')

from lib import dataset
from lib import util

classify = False
bin_class = False
small_obj_heights = False

def norm_predictions(preds):
    max_pred = 0
    min_pred = 1000
    normed = [None] * len(preds)
    idx = 0

    for pred in preds:
        if pred < min_pred:
            min_pred = pred
        if pred > max_pred:
            max_pred = pred


    for pred in preds:
        p = (pred-min_pred)/(max_pred - min_pred)
        normed[idx] = p
        idx = idx + 1


    return normed


def renderPredictions(dataset, predictions, preResu):
    heightpreds = {}
    count = 0
    for pt, pred in predictions.items():
        locName, location = pt
        loc = locName
        if loc not in heightpreds:
            heightpreds[loc] = []

        if classify:
            if count == 0:
                pair = (location, 0)
                count = count + 1
            else:
                pair = (location, (np.argmax(pred) + 1))
        else:
            pair = (location, float(pred))

        heightpreds[loc].append(pair)


    results = {}
    for locName, locsAndPreds in heightpreds.items():
        locs, preds = zip(*locsAndPreds)
        xs,ys = zip(*locs)
        norm_pred = norm_predictions(preds)
        preds = [pred+1 for pred in norm_pred]
        loc = dataset.data.locs[locName]
        canvas = np.zeros(loc.layerSize, dtype=np.float32)
        canvas[(xs,ys)] = np.array(preds, dtype=np.float32)

        results[locName] = canvas
    return results

def get_error(val_height, pred):
    return abs(val_height - pred)

def get_allowable_error(val_height):

    if small_obj_heights:
        four = 4/150
        return_small = 1/150
        return_tall = (.25 * val_height)/150
    else:
        four = 4
        return_small = 1
        return_tall = (.25 * val_height)

    if val_height < four:
        return return_small
    else:
        return return_tall

def evaluate_heights(predictions, dataset):
    current_locName = ""

    underTwo_acc = 0
    underTwo_total = 0

    twoFive_acc = 0
    twoFive_total = 0

    fiveTwenty_acc = 0
    fiveTwenty_total = 0

    twentyFifty_acc = 0
    twentyFifty_total = 0

    fifty75_acc = 0
    fifty75_total = 0
    fiftyPlus_acc = 0
    fiftyPlus_total = 0

    seven5Hund_acc = 0
    seven5Hund_total = 0

    hundPlus_acc = 0
    hundPlus_total = 0

    underFive_total = 0
    underFive_acc = 0

    fiveTen_acc = 0
    tenTwenty_acc = 0
    twentyThirty_acc = 0
    thirtyFourty_acc = 0
    fourtyFifty_acc = 0


    fiveTen_total = 0
    tenTwenty_total = 0
    twentyThirty_total = 0
    thirtyFourty_total = 0
    fourtyFifty_total = 0

    acc = 0
    not_acc = 0
    location_names = []
    total_error = 0

    total_pred = len(predictions)


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

    count_it = 0

    for pt, pred in predictions.items():
        locName, location = pt

        if locName != current_locName:
            loc = dataset.data.locs[locName]
            current_locName = locName
            location_names.append(locName)


        if classify:
            if bin_class:
                count_it = count_it + 1
                val_height_arr = loc.obj_height_classification[location]
                val_height = np.argmax(val_height_arr) + 1
                pred_height = np.argmax(pred) + 1

                if val_height == 1:
                    underFive_total += 1
                else:                               #this is 5-20
                    fiveTwenty_total += 1

                if val_height == pred_height:
                    acc = acc + 1
                    if val_height == 1:
                        underFive_acc += 1
                    else:
                        fiveTwenty_acc += 1
                else:
                    not_acc = not_acc + 1
            else:
                val_height_arr = loc.obj_height_classification[location]
                val_height = np.argmax(val_height_arr)

                pred_height = np.argmax(pred)

                if val_height == 0:
                    underFive_total += 1
                elif val_height == 1:                               #this is 5-20
                    fiveTwenty_total += 1
                elif val_height == 2:                               #this is 20-50
                    twentyFifty_total += 1
                else:
                    fiftyPlus_total += 1

                if val_height == pred_height:
                    acc = acc + 1
                    if val_height == 0:
                        underFive_acc += 1
                    elif val_height == 1:
                        fiveTwenty_acc += 1
                    elif val_height == 2:
                        twentyFifty_acc += 1
                    else:
                        fiftyPlus_acc += 1
                else:
                    not_acc = not_acc + 1
        else:
            val_height = loc.layer_obj_heights[location]
            error = get_error(val_height, pred)
            total_error = total_error + error
            allowable_error = get_allowable_error(val_height)

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

            # print(val_height)

            if val_height < two:
                underTwo_total = underTwo_total + 1
            elif val_height < five:
                twoFive_total = twoFive_total + 1
            elif val_height < ten:
                fiveTen_total = fiveTen_total + 1
            elif val_height < twenty:
                tenTwenty_total = tenTwenty_total + 1
            elif val_height < thirty:
                twentyThirty_total = twentyThirty_total + 1
            elif val_height < fourty:
                thirtyFourty_total = thirtyFourty_total + 1
            elif val_height < fifty:
                fourtyFifty_total = fourtyFifty_total + 1
            elif val_height < seven_five:
                fifty75_total = fifty75_total + 1
            elif val_height < hund:
                seven5Hund_total = seven5Hund_total + 1
            else:
                hundPlus_total = hundPlus_total + 1

            if error < allowable_error:
                acc = acc + 1
                if val_height < two:
                    underTwo_acc = underTwo_acc + 1
                elif val_height < five:
                    twoFive_acc = twoFive_acc + 1
                elif val_height < ten:
                    fiveTen_acc = fiveTen_acc + 1
                elif val_height < twenty:
                    tenTwenty_acc = tenTwenty_acc + 1
                elif val_height < thirty:
                    twentyThirty_acc = twentyThirty_acc + 1
                elif val_height < fourty:
                    thirtyFourty_acc = thirtyFourty_acc + 1
                elif val_height < fifty:
                    fourtyFifty_acc = fourtyFifty_acc + 1
                elif val_height < seven_five:
                    fifty75_acc = fifty75_acc + 1
                elif val_height < hund:
                    seven5Hund_acc = seven5Hund_acc + 1
                else:
                    hundPlus_acc = hundPlus_acc + 1
            else:
                not_acc = not_acc + 1

    if classify:
        if bin_class:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print('Percent < 10 accurate:       ', underFive_acc/underFive_total)
            print('Percent x >= 10 accurate:    ', fiveTwenty_acc/fiveTwenty_total)
            print('Total accurate:              ', acc)
            print('Total not accurate:          ', not_acc)

            results = {'perc_accurate': acc/total_pred,
                        'avg_error_feet': total_error/total_pred,
                        'percent_acc_under10': underFive_acc/underFive_total,
                        'percent_acc_over10': fiveTwenty_acc/fiveTwenty_total,
                        'total_accurate': acc,
                        'total_not_acc': not_acc,
                        'locations': location_names}
        else:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print('Percent < 5 accurate:       ', underFive_acc/underFive_total)
            print('Percent 5 < x < 20 accurate:', fiveTwenty_acc/fiveTwenty_total)
            print('Percent 20 < x < 50 accurate:', twentyFifty_acc/twentyFifty_total)
            print('Percent 50 <= accurate:       ', fiftyPlus_acc/fiftyPlus_total)
            print('Total accurate:              ', acc)
            print('Total not accurate:          ', not_acc)

            results = {'perc_accurate': acc/total_pred,
                        'avg_error_feet': total_error/total_pred,
                        'percent_acc_under5': underFive_acc/underFive_total,
                        'percent_acc_5-20': fiveTwenty_acc/fiveTwenty_total,
                        'percent_acc_20-50': twentyFifty_acc/twentyFifty_total,
                        'percent_acc_50plus': fiftyPlus_acc/fiftyPlus_total,
                        'total_accurate': acc,
                        'total_not_acc': not_acc,
                        'locations': location_names}
    else:
        try:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print('Percent < 2 accurate:       ', underTwo_acc/underTwo_total)
            print('Percent 2 < x < 5 accurate: ', twoFive_acc/twoFive_total)
            print('Percent 5 < x < 10 accurate:', fiveTen_acc/fiveTen_total)
            print('Percent 10 < x < 20 accurate:', tenTwenty_acc/tenTwenty_total)
            print('Percent 20 < x < 30 accurate:', twentyThirty_acc/twentyThirty_total)
            print('Percent 30 < x < 40 accurate:', thirtyFourty_acc/thirtyFourty_total)
            print('Percent 40 < x < 50 accurate:', fourtyFifty_acc/fourtyFifty_total)
            print('Percent 50 < x < 75 accurate:', fifty75_acc/fifty75_total)
            print('Percent 75 < x <100 accurate:', seven5Hund_acc/seven5Hund_total)
            print('Percent 100 < accurate:      ', hundPlus_acc/hundPlus_total)
            print('Total accurate:              ', acc)
            print('Total not accurate:          ', not_acc)
        except:
            print('Percent accurate:           ', acc/total_pred)
            print('Average error in feet:      ', total_error/total_pred)
            print("There was a division by zero")

        results = {'perc_accurate': acc/total_pred,
                    'avg_error_feet': total_error/total_pred,
                    'percent_acc_under2': underTwo_acc/acc,
                    'percent_acc_2-5': twoFive_acc/acc,
                    'percent_acc_5-10': fiveTen_acc/acc,
                    'percent_acc_10-20': tenTwenty_acc/acc,
                    'percent_acc_20-30': twentyThirty_acc/acc,
                    'percent_acc_30-40': thirtyFourty_acc/acc,
                    'percent_acc_40-50': fourtyFifty_acc/acc,
                    'percent_acc_50-75': fifty75_acc/acc,
                    'percent_acc_75-100': seven5Hund_acc/acc,
                    'percent_acc_100plus': hundPlus_acc/acc,
                    'total_accurate': acc,
                    'total_not_acc': not_acc,
                    'locations': location_names}

    return results



def createCanvases(dataset):
    result = {}
    for locName in dataset.getUsedLocNames():
        print(locName)
        loc = dataset.data.locs[locName]
        l = loc.loadVeg(loc.name)
        h,w = l.shape
        normedDEM = util.normalize(loc.loadVeg2())

        canvas = cv2.cvtColor(normedDEM, cv2.COLOR_GRAY2RGB)
        im2, startContour, hierarchy = cv2.findContours(l.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im2, endContour, heirarchy = cv2.findContours(l.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result[locName] = canvas
    return result

def overlay(predictionRenders, canvases):
    result = {}
    for locName in sorted(canvases):
        canvas = canvases[locName].copy()
        render = predictionRenders[locName]
        yellowToRed = np.dstack((np.ones_like(render), 1-(render-1), np.zeros_like(render)))
        canvas[render>1] = yellowToRed[render>1]
        result[locName] = canvas
    return result

def visualizePredictions(dataset, predictions, preResu):
    predRenders = renderPredictions(dataset, predictions, preResu)
    canvases = createCanvases(dataset)
    overlayed = overlay(predRenders, canvases)
    return overlayed

def showPredictions(predictionsRenders):
    locs = {}
    for locName, render in predictionsRenders.items():
        if locName not in locs:
            locs[locName] = []
        locs[locName].append(render)

    for locName, frameList in locs.items():
        frameList.sort()
        fig = plt.figure(locName, figsize=(8, 6))
        ims = []
        pos = (30,30)
        color = (0,0,1.0)
        size = 1
        thickness = 2
        for render in frameList:
            withTitle = render.copy()
            cv2.putText(render, locName, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness=thickness)
            im = plt.imshow(render)
            ims.append([im])
        anim = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                repeat_delay=0)

        def createMyOnKey(anim):
            def onKey(event):
                if event.key == 'right':
                    anim._step()
                elif event.key == 'left':
                    saved = anim._draw_next_frame
                    def dummy(a,b):
                        pass
                    anim._draw_next_frame = dummy
                    for i in range(len(anim._framedata)-2):
                        anim._step()
                    anim._draw_next_frame = saved
                    anim._step()
                elif event.key =='down':
                    anim.event_source.stop()
                elif event.key =='up':
                    anim.event_source.start()
            return onKey

        fig.canvas.mpl_connect('key_press_event', createMyOnKey(anim))
        plt.show()


def show(*imgs, imm=True):
    try:
        for i, img in enumerate(imgs):
            plt.figure(i, figsize=(8, 6))
            plt.imshow(img)
        if imm:
            plt.show()
    except:
        print("Not able to show because plt not imported")

def save(img, name):
    fname = 'output/imgs/{}.png'.format(name)
    cv2.imwrite(fname, img)

def saveModel(model, mod_string):
    from keras.utils import plot_model
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}_{}.png'.format(timeString, mod_string)
    plot_model(model, to_file=fname, show_shapes=True)
