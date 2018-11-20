import pandas as pd
import numpy as np

dataSet = pd.read_csv("train.csv")

def initLoad():
    """Loading the data from the dataSet ages, gender, classNum"""
    ages = np.array(dataSet['Age'], dtype=np.float16)
    ages = ages.reshape((ages.size, 1))
    ages = isnan(ages)
    #ages = Scale(ages)

    gender = np.array(dataSet['Sex'])
    gender = gender.reshape((gender.size, 1))
    gender = Gender(gender)

    classNum = np.array(dataSet['Pclass'])
    classNum = classNum.reshape((classNum.size, 1))
    #classNum = Scale(classNum)

    m = len(dataSet)    # size of dataSet
    Y = np.array(dataSet['Survived'])
    Y = Y.reshape(Y.size, 1)
    X0 = np.ones( (m, 1) )
    fet = np.stack((X0, ages, classNum, gender))
    return fet, Y

def Gender(gender):
    for i in range(len(gender)):
        gender[i] = 1 if gender[i] == 'male' else 0
    return gender

def isnan(F):
    s = len(F)
    for i in range(s):
        if np.isnan(F[i]):
            F[i] = np.nanmean(F)
    return F

def Scale(featureX):
    scaled = np.array([])
    scaled = (featureX - featureX.mean()) / featureX.size
    return scaled

def decision(yPredicted):
    if yPredicted >= 0.5:
        return 1
    else:
        return 0


def hyposisFN(thetas, features):
    z = np.array(np.dot(thetas, features), dtype=np.float16)
    pre = np.array(1 / (1 + np.exp(-z)), dtype=np.float16)
    return pre


def costFN(thetas, features, Y):
    m = len(features)
    return (-1 * sum(Y * np.log(hyposisFN(thetas.transpose(), features)) + (1 + Y) * np.log(1 - hyposisFN(thetas.transpose(), features))) / m)


def gradientDescent(thetas, features, Y, alpha, iterations):  # Victorized Implementation
    m = len(features)
    for i in range(iterations):
        val = (hyposisFN(thetas.transpose(), features) - Y)
        #thetas = thetas - (alpha/m )* (np.dot(features, val))
    return val.shape


def main():
    fet, y = initLoad()
    thetas = np.zeros((fet.shape[0], 1))
    fet = fet.reshape((4, 891))
   # pr = hyposisFN(thetas.transpose(), fet)
    cost = gradientDescent(thetas, fet, y, 0.1, 150)
    print(cost)
    #print(pr)


if __name__ == '__main__':
    main()
