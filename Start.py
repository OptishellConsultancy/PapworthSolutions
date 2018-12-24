#
from readxlsmData import ReadData
from SimpleMLPTrainer import SimpleMLPTrainer
import numpy as np
searchColumns = ["Sex","LC","Height","Age","Weight","Sex","LC","Height","Age","Weight","QofLife","Survival"]

#Reading data
RD = ReadData('data/AISample.xlsm', 5, 34)
dataDonorsSet  = []
dataRecipientSet = []
dataRequiredResultSet  = []
#Print data and replace M/F with 0 and 1 respectively
row = 0
for rows in RD.Data:
    cell = 0    
    dataDonorsSet.append([])
    dataRecipientSet.append([])
    dataRequiredResultSet.append([])

    for cells in rows:
    #Replacing all values as floatt types and extracting data for organization best for minimization
        if RD.Data[row][cell] == 'M':
            RD.Data[row][cell] = 1.0
        if RD.Data[row][cell] == 'F':
            RD.Data[row][cell] = -1.0 
        RD.Data[row][cell] = float(1.0/(RD.Data[row][cell]+0.000001)) #Normalize and cast to float with epsilon
        print("row: {}, Cell: {}, Value: {}".format(row,cell,RD.Data[row][cell])) 
        #Extract data 

        if cell < 5:
            dataDonorsSet[row].append(RD.Data[row][cell])
        if cell >= 5 and cell < 10 :
            dataRecipientSet[row].append(RD.Data[row][cell])
        if cell >= 10 and cell <= 11 :
            dataRequiredResultSet[row].append(RD.Data[row][cell])
     
        cell += 1
    row += 1   


SMT = SimpleMLPTrainer(dataDonorsSet ,dataRecipientSet ,dataRequiredResultSet,row)