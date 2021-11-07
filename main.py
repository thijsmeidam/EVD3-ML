# This is a sample Python script.
import cv2 as cv
import numpy as np
import os
from sklearn.utils import Bunch
import glob
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def getLargestContour(img_BW):
    """ Return largest contour in foreground as an nd.array """
    contours, hier = cv.findContours(img_BW.copy(), cv.RETR_TREE,
                                     cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv.contourArea)

    return np.squeeze(contour)


def getContourExtremes(contour):
    """ Return contour extremes as an tuple of 4 tuples """
    # determine the most extreme points along the contour
    left = contour[contour[:, 0].argmin()]
    right = contour[contour[:, 0].argmax()]
    top = contour[contour[:, 1].argmin()]
    bottom = contour[contour[:, 1].argmax()]

    return np.array((left, right, top, bottom))


def getConvexityDefects(contour):
    """ Return convexity defects in a contour as an nd.array """
    hull = cv.convexHull(contour, returnPoints=False)
    hull[::-1].sort(axis=0)
    defects = cv.convexityDefects(contour, hull)
    if defects is not None:
        defects = defects.squeeze()

    return defects


def getSimpleContourFeatures(contour):
    """ Return some simple contour features
        See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
    """
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area
    features = np.array((area, perimeter, aspect_ratio, extent))

    return (features)


def getContourFeatures(contour):
    """ Return some contour features
    """
    # basic contour features
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    extremePoints = getContourExtremes(contour)

    # get contour convexity defect depths
    # see https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    defects = getConvexityDefects(contour)

    defect_depths = defects[:, -1] / 256.0 if defects is not None else np.zeros((4,1))

    # select only the 6 largest depths
    defect_depths = np.flip(np.sort(defect_depths))[0:4]
    if(len(defect_depths)<4):
        x=4-len(defect_depths)
        defect_depths=np.append(defect_depths,np.zeros(x))
    # compile a feature vector
    features = np.append(defect_depths,(area, perimeter))
    return (features)
def fetch_data(data_path,data_path2,data_path3,data_path4):
    # grab the list of images in our data directory
    mylist=[data_path,data_path2,data_path3,data_path4]
    feature_names = ['depths','depth2','depth3','depth4','area', 'perimeter']
    data = np.empty((0, len(feature_names)), float)
    target = []
    for listname in mylist:
        p = os.path.sep.join([listname, '**', '*.jpg'])

        file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]

        # intitialize data matrix with correct number of features


        # loop over the image paths
        for filename in file_list:  # [::10]:
            # load image and blur a bit to suppress noise
            img = cv.imread(filename)
            img = cv.GaussianBlur(img, (3, 3), 0, dst=None)
            #cv.imshow("image2", img)
            cann = cv.Canny(img, 120, 180, None, 3, True)
            cv.imshow("image3", cann)
            cann = cv.morphologyEx(cann, cv.MORPH_GRADIENT, None, iterations=1)
            cv.imshow("gradient", cann)
            contours=getLargestContour(cann)

            # mask background

            # perform a series of erosions and dilations to remove any small regions of noise


            # check if foreground is actually there
            if cv.countNonZero(cann) == 0:
                continue

            #cv.imshow("Segmented image", cann)

            # find largest contour
            contour = getLargestContour(cann)
            features = getContourFeatures(contours)
            # extract features from contour
            getSimpleContourFeatures(contour)
            for x in range(6):
                if features[x]==None:
                    features[x]= np.zeros(1)

            # extract label from folder name and stor
            label = filename.split(os.path.sep)[-2]
            target.append(listname)

            # append features to data matrix
            data = np.append(data, np.array([features]), axis=0)

            # draw outline, show image, and wait a bit
            cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
            cv.imshow("image", img)

        unique_targets = np.unique(target)

        dataset = Bunch(data=data,
                        target=target,
                        unique_targets=unique_targets,
                        feature_names=feature_names)

    return dataset

def fetch_data_solo(data_path):
    # grab the list of images in our data directory
    mylist=[data_path]
    feature_names = ['depths','depth2','depth3','depth4','area', 'perimeter']
    data = np.empty((0, len(feature_names)), float)
    target = []
    for listname in mylist:
        p = os.path.sep.join([listname, '**', '*.jpg'])

        file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
        # intitialize data matrix with correct number of features


        # loop over the image paths
        for filename in file_list:  # [::10]:
            # load image and blur a bit to suppress noise
            img = cv.imread(filename)
            img = cv.GaussianBlur(img, (3, 3), 0, dst=None)
            #cv.imshow("image2", img)
            cann = cv.Canny(img, 120, 180, None, 3, True)
            cv.imshow("image3", cann)
            cann = cv.morphologyEx(cann, cv.MORPH_GRADIENT, None, iterations=1)
            cv.imshow("gradient", cann)
            contours=getLargestContour(cann)

            # mask background

            # perform a series of erosions and dilations to remove any small regions of noise


            # check if foreground is actually there
            if cv.countNonZero(cann) == 0:
                continue

            #cv.imshow("Segmented image", cann)

            # find largest contour
            contour = getLargestContour(cann)
            features = getContourFeatures(contours)
            # extract features from contour
            getSimpleContourFeatures(contour)
            for x in range(6):
                if features[x]==None:
                    features[x]= np.zeros(1)

            # extract label from folder name and stor
            label = filename.split(os.path.sep)[-2]
            target.append(listname)

            # append features to data matrix
            data = np.append(data, np.array([features]), axis=0)

            # draw outline, show image, and wait a bit
            cv.drawContours(img, [contour], -1, (0, 255, 0), 2)
            cv.imshow("image", img)

        unique_targets = np.unique(target)

        dataset = Bunch(data=data)

    return data
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    while True:
        currentPath=os.getcwd()
        data_path = currentPath+"/training A/"
        data_path2 = currentPath+"/training B/"
        data_path3 = currentPath+"/training c/"
        data_path4 = currentPath+"/training d/"
        data_path5 = currentPath+"/testing/"

        # fetch the data
        gestures = fetch_data(data_path,data_path2,data_path3,data_path4)
        # encode the categorical labels
        le = LabelEncoder()
        coded_labels = le.fit_transform(gestures.target)

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (trainX, testX, trainY, testY) = train_test_split(gestures.data, coded_labels,test_size=0.4,train_size=0.6,stratify=gestures.target)  # , random_state=42)
        # Data preparation (note that a pipeline  would help here
        # show target distribution
        ax = sns.countplot(x=trainY, color="skyblue")
        ax.set_xticklabels(gestures.unique_targets)
        ax.set_title('picture count')
        plt.tight_layout()
        # show histograms of first 4 features
        fig0, ax0 = plt.subplots(2, 2)
        sns.histplot(trainX[:, 0], color="skyblue", bins=10, ax=ax0[0, 0])
        sns.histplot(trainX[:, 1], color="olive", bins=10, ax=ax0[0, 1])  # , axlabel=gestures.feature_names[1])
        sns.histplot(trainX[:, 2], color="gold", bins=10, ax=ax0[1, 0])  # , axlabel=gestures.feature_names[2])
        sns.histplot(trainX[:, 3], color="teal", bins=10, ax=ax0[1, 1])  # , axlabel=gestures.feature_names[3])
        ax0[0, 0].set_xlabel(gestures.feature_names[0])
        ax0[0, 1].set_xlabel(gestures.feature_names[1])
        ax0[1, 0].set_xlabel(gestures.feature_names[2])
        ax0[1, 1].set_xlabel(gestures.feature_names[3])
        plt.tight_layout()

        # show scatter plot of features a and b
        a, b = 2, 5
        fig1 = plt.figure()
        ax1 = sns.scatterplot(trainX[:, a], trainX[:, b], hue=le.inverse_transform(trainY))
        ax1.set_title("Example of feature scatter plot")
        ax1.set_xlabel(gestures.feature_names[a])
        ax1.set_ylabel(gestures.feature_names[b])
        plt.tight_layout()
        a, b = 3, 5
        fig1 = plt.figure()
        ax1 = sns.scatterplot(trainX[:, a], trainX[:, b], hue=le.inverse_transform(trainY))
        ax1.set_title("Example of feature scatter plot")
        ax1.set_xlabel(gestures.feature_names[a])
        ax1.set_ylabel(gestures.feature_names[b])
        plt.tight_layout()
        a, b = 1, 5
        fig1 = plt.figure()
        ax1 = sns.scatterplot(trainX[:, a], trainX[:, b], hue=le.inverse_transform(trainY))
        ax1.set_title("Example of feature scatter plot")
        ax1.set_xlabel(gestures.feature_names[a])
        ax1.set_ylabel(gestures.feature_names[b])
        plt.tight_layout()

        ##    # show joint distribution plot of features a and b for 2 selected labels
        ##    a, b = 0, 1
        ##    c, d = le.transform(['paper', 'rock'])
        ##    sns.set_style("whitegrid")
        ##    indices = np.where( (trainY==c) | (trainY==d))
        ##    ax2 = sns.jointplot(x=trainX[indices,a], y=trainX[indices,b], kind="kde")
        ##    ax2.set_axis_labels(gestures.feature_names[a], gestures.feature_names[b])
        ##    plt.tight_layout()

        # show boxplot for a single feature
        a = 0
        plt.figure()
        ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:, a])
        ax3.set_title(gestures.feature_names[a])
        ax3.set_ylabel(gestures.feature_names[a])
        plt.tight_layout()
        a = 1
        plt.figure()
        ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:, a])
        ax3.set_title(gestures.feature_names[a])
        ax3.set_ylabel(gestures.feature_names[a])
        plt.tight_layout()
        a = 2
        plt.figure()
        ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:, a])
        ax3.set_title(gestures.feature_names[a])
        ax3.set_ylabel(gestures.feature_names[a])
        plt.tight_layout()
        a = 3
        plt.figure()
        ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:, a])
        ax3.set_title(gestures.feature_names[a])
        ax3.set_ylabel(gestures.feature_names[a])
        plt.tight_layout()
        a = 4
        plt.figure()
        ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:, a])
        ax3.set_title(gestures.feature_names[a])
        ax3.set_ylabel(gestures.feature_names[a])
        plt.tight_layout()
        a = 5
        plt.figure()
        ax3 = sns.boxplot(x=le.inverse_transform(trainY), y=trainX[:, a])
        ax3.set_title(gestures.feature_names[a])
        ax3.set_ylabel(gestures.feature_names[a])
        plt.tight_layout()
        # show feature correlation heatmap
        plt.figure()
        corr = np.corrcoef(trainX, rowvar=False)
        ax4 = sns.heatmap(corr, annot=True, xticklabels=gestures.feature_names, yticklabels=gestures.feature_names)
        plt.tight_layout()

        plt.show(block=False)


        iris_dataframe = pd.DataFrame(trainX, columns=gestures.feature_names)
        # create a scatter matrix from the dataframe, color by y_train
        pd.plotting.scatter_matrix(iris_dataframe, c=trainY, figsize=(15, 15), marker='o', hist_kwds={'bins': 20},
                                   s=60, alpha=.8)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(trainX, trainY)
        # print(knn)
        # plt.show()
        # plt.show()
        ConfusionMatrixDisplay.from_estimator(knn,testX,testY)
        plt.show()
        test=fetch_data_solo(data_path5)
        prediction=knn.predict(test)
        print("predictions:")
        print(prediction)
        score = cross_val_score(knn, testX, testY, cv=3, scoring="recall_macro")
        print(score.mean())
        score = cross_val_score(knn, testX, testY, cv=3, scoring="precision_macro")
        print(score.mean())
        score = cross_val_score(knn, testX, testY, cv=3, scoring="accuracy")
        print(score.mean())
        input("hello")

        break
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
