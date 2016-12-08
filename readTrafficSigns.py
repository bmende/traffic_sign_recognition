# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
import csv

path_to_gtsrb = "./GTSRB/Final_Training/Images"
path_to_test = "./GTSRB/Final_Test/Images/"

HEIGHT = 36
WIDTH = 36

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath=path_to_gtsrb):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        count = 0
        for row in gtReader:
            image = plt.imread(prefix + row[0])
            image = resize(image, [HEIGHT, WIDTH])
            images.append(image) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
            # count += 1
            # if count > 10:
            #     break
        gtFile.close()
    return images, labels

def readTrafficSigns_test(rootpath=path_to_test):

    images = []
    labels = []
    prefix = rootpath + '/'
    gtFile = open(prefix + 'GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header

    count = 0
    for row in gtReader:
        image = plt.imread(prefix + row[0])
        image = resize(image, [HEIGHT, WIDTH])
        images.append(image)
        labels.append(row[7])
        # count += 1
        # if count > 10:
        #     break

    gtFile.close()

    return images, labels



if __name__ == "__main__":


    print "getting files"
    images, labels = readTrafficSigns(path_to_gtsrb)
    print "files got"
