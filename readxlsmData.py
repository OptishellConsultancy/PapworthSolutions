from openpyxl import Workbook
from openpyxl import load_workbook
import copy
class ReadData(object):    
    Data = []
    ColumnHeaderMap = []
    def __init__(self,file,startRowInd,endRowInd):        
        wb = load_workbook(filename=file, data_only=True )
        ws = wb.worksheets[0]

        print("Reading Data")
        #First get all data
        RowIndex = 0
        RowsAddedCount = 0      
        for row in ws.rows:
            if RowIndex >= startRowInd and RowIndex <= endRowInd:
                #print("Row: {} ".format(RowIndex))
                self.Data.append([])                   
                for cell in row:       
                    self.Data[RowsAddedCount].append(cell._value)
                    print("Added value: {} ".format(cell._value))        
                RowsAddedCount += 1
            RowIndex += 1
            #
        
   