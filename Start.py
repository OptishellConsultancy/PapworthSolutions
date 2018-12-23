#
from readxlsmData import ReadData
from SimpleMLPTrainer import SimpleMLPTrainer
import numpy as np
searchColumns = ["Sex","LC","Height","Age","Weight","Sex","LC","Height","Age","Weight","QofLife","Survival"]

#Reading data
RD = ReadData('data/AISample.xlsm', 5, 34)

row = 0
for rows in RD.Data:
    cell = 0    
    for cells in rows:
    #Now rearrange for entry as a row
        print("row: {}, Cell: {}, Value: {}".format(row,cell,RD.Data[row][cell]))    
        cell += 1
    row += 1   

#Now reorganize data for training:

#Uncomment each test as needed:
#SMT = SimpleMLPTrainer(RD.ReadValidData,9,11)

    