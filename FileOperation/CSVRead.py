import csv
import numpy as np

def readCSVByRow(filePathAndFileName):
    resultList = []
    with open(filePathAndFileName + '.csv', newline='') as csvfile:
        dataList = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in dataList:
            one = []
            for i in range(0, len(row)):
                one.append(int(float(row[i])))
            resultList.append(one)
    return np.array(resultList)