import matplotlib.pyplot as plt
import numpy
import glob
import os
import time

from PIL import Image
from skimage.feature import hog
from skimage import exposure
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def GetFileList(folder):
    fileList = []
    for imagePath in glob.glob(os.path.join(folder, '*.jpg')):
        fileList.append(imagePath)
    return fileList

def ReadImage(inName):
    return Image.open(inName)

def SaveImage(image, outName):
    return image.save(outName)

def ResizeImage(inImage, nWidth, nHeight):
    return inImage.resize((nWidth, nHeight), Image.ANTIALIAS)

def GetFeatureByHog(inImage, ppc, cpb, bOutputHogImage = False):
    
    image = numpy.array(inImage.getdata(),
        numpy.uint8).reshape(inImage.size[1], inImage.size[0], 3)
        
    image = exposure.adjust_gamma(image, gamma=0.5)
    
    ppcT = (ppc, ppc)
    cpbT = (cpb, cpb)
    
    if bOutputHogImage == False:
        fd = hog(image, orientations=8, pixels_per_cell=ppcT, block_norm = 'L1',
            cells_per_block=cpbT, visualize=False, multichannel=True)
    else:
        fd, hog_image= hog(image, orientations=8, pixels_per_cell=ppcT, block_norm = 'L1',
            cells_per_block=cpbT, visualize=True, multichannel=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    return fd

class DataPack:
    pass
    
def GetFeature(fileList, label, resize, w, h, ppc, cpb):
    num = 0
    dataPack = DataPack()
    dataPack.sampleVector = []
    dataPack.labelVector = []

    for imagePath in fileList:
        num += 1
        image = ReadImage(imagePath)
        if resize == True:
            image = ResizeImage(image, w, h)
        featureVector = GetFeatureByHog(image, ppc, cpb, False)
        dataPack.sampleVector.append(featureVector)
        dataPack.labelVector.append(label)
        if num % 100 == 0:
            print('.', end='', flush=True)

    print("%d sample processed" %num)

    return dataPack

def GetAllFeature(folder, resize, w, h, ppc, cpb):
    if resize == True:
        subFolder = "org/"
    else:
        subFolder = str(w)
        
    dataFolderCat = folder + "cat/" + subFolder
    dataFolderDog = folder + "dog/" + subFolder
    
    fileListCat = GetFileList(dataFolderCat)
    fileListDog = GetFileList(dataFolderDog)
    
    fileListCatTrain, fileListCatTest, fileListDogTrain, fileListDogTest = train_test_split(fileListDog, fileListCat, test_size=0.1, train_size=0.1)
    
    dataPackTrain = DataPack()
    dataPackTrain.sampleVector = []
    dataPackTrain.labelVector = []

    dataPackTest = DataPack()
    dataPackTest.sampleVector = []
    dataPackTest.labelVector = []

    t0 = time.time()
    
    subDataPack = GetFeature(fileListCatTrain, 0, resize, w, h, ppc, cpb)
    dataPackTrain.sampleVector.extend(subDataPack.sampleVector)
    dataPackTrain.labelVector.extend(subDataPack.labelVector)

    subDataPack = GetFeature(fileListDogTrain, 1, resize, w, h, ppc, cpb)
    dataPackTrain.sampleVector.extend(subDataPack.sampleVector)
    dataPackTrain.labelVector.extend(subDataPack.labelVector)
    
    subDataPack = GetFeature(fileListCatTest, 0, resize, w, h, ppc, cpb)
    dataPackTest.sampleVector.extend(subDataPack.sampleVector)
    dataPackTest.labelVector.extend(subDataPack.labelVector)

    subDataPack = GetFeature(fileListDogTest, 1, resize, w, h, ppc, cpb)
    dataPackTest.sampleVector.extend(subDataPack.sampleVector)
    dataPackTest.labelVector.extend(subDataPack.labelVector)
    
    t1 = time.time()
    print('Prepare data: %f seconds' % (t1-t0))
    
    return dataPackTrain, dataPackTest

def main():
    print("Start")

    dataPackTrain, dataPackTest = GetAllFeature('../data/', False, 128, 128, 16, 1)
    dataPackTrain.sampleVector = normalize(dataPackTrain.sampleVector)
    dataPackTest.sampleVector = normalize(dataPackTest.sampleVector)
    
    print("Feature num: %d" %(len(dataPackTrain.sampleVector[0])))

    t0 = time.time()

    clf = svm.LinearSVC(verbose=1)
    #clf = svm.SVC(kernel='linear', cache_size=2000, verbose=True)
    #clf = svm.SVC(cache_size=2000, verbose=True)
    clf.fit(dataPackTrain.sampleVector, dataPackTrain.labelVector)

    t1 = time.time()
    print('Training time: %f seconds' % (t1-t0))

    t0 = time.time()
    acc = clf.score(dataPackTest.sampleVector, dataPackTest.labelVector)
    t1 = time.time()
    print('Predicting data: %f seconds' % (t1-t0))
    
    print('Accurancy: %.1f%%' %(acc*100))

if __name__ == "__main__":
    main()
