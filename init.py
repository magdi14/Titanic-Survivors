import pandas as pd
import numpy as np

dataSet = pd.read_csv("train.csv")

def initLoad():
    """Loading the data from the dataSet ages, gender, classNum"""
    ages = np.array(dataSet['Age'], dtype=np.float64)
    ages = ages.reshape((ages.size, 1))
    ages = isnan(ages)
    #ages = Scale(ages)

    gender = np.array(dataSet['Sex'], dtype=np.float64)
    gender = gender.reshape((gender.size, 1))
    gender = Gender(gender)

    classNum = np.array(dataSet['Pclass'], dtype=np.float64)
    classNum = classNum.reshape((classNum.size, 1))
    #classNum = Scale(classNum)

    m = len(dataSet)    # size of dataSet
    Y = np.array(dataSet['Survived'])
    Y = Y.reshape(Y.size, 1)
    X0 = np.ones( (m, 1) )
    fet = np.stack((X0, ages, classNum, gender))
    fet = fet.reshape((4, 891))
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

def predictedY(thetas, features):
    y = np.array(list((map(decision, hyposisFN(theta, features)))))
    return y


def hyposisFN(thetas, features):
    z = np.array(np.dot(thetas, features), dtype=np.float64)
    pre = np.array(1 / (1 + np.exp(-z)), dtype=np.float64)
    return pre


def costFN(thetas, features, Y):
    m = len(features)
    return -1 / m * (sum(Y * np.log(hyposisFN(thetas, features)) + (1-Y) * np.log(1 - hyposisFN(thetas, features))))

def gradientDescent(thetas, features, Y, alpha, iterations):        #Victorized Implementation
    m = len(Y)
    for i in range(iterations):
        derv = (hyposisFN(thetas, features) - Y)
        thetas = thetas - ((alpha/m) * np.dot(features, derv.transpose()))

    return thetas

def cnt(Y):
    sz = len(Y)
    z = one = 0
    for i in range(sz):
        if Y[i] == 0:
            z+=1
        else:
            one+=1
    return z, one

def main():
    fet, y = initLoad()
    thetas = np.zeros((1, fet.shape[0]))

    print(thetas.shape, fet.shape, y.shape)
    pr = hyposisFN(thetas, fet)
    thetas = gradientDescent(thetas, fet, y.transpose(), 0.1, 600)
    #ypre = predictedY(thetas, fet)
    print(thetas)
    #print(y)
    #z, on = cnt(y)
    #print(z, on)
    print(thetas.shape, fet.shape, y.shape)
    #cost = costFN(thetas, fet, y)

    #print(cost)
    #print(pr)


if __name__ == '__main__':
    main()
