import numpy as np
import pandas as pd
import datetime
#start:
#Path to Original Training Dataset "view _data" File
dataBefore = "/content/GRU4REC-pytorch/data/raw_data/event2.csv"
 #Path to Processed Dataset Folder
dataAfter = "/content/GRU4REC-pytorch/data/preprocessed_data"
#End:

#I DONT NEED USER THAT EXIST ONLY 1 TIME SO HERE REMOVE VISITORS OF LESS THAN 2 INTERACTIONS
def removeShortSessions(data):
    #delete USER of length <= 1  
    sessionLen = data.groupby('visitorid').size() #group by visitorid and get size of each visitor
    data = data[np.in1d(data.visitorid, sessionLen[sessionLen > 1].index)]#هنا بتقوله هات الاكبر من واحد 
    return data


#Read Dataset in pandas Dataframe (Ignore Category Column)
train = pd.read_csv(dataBefore)
train.columns = ['visitorid', 'itemid', 'date'] #Headers of dataframe

#start:
train['date'] = pd.to_datetime(train['date'] , unit = 'ms')
train['date'] = pd.to_datetime(train['date'] , format = '%Y-%m-%dT%H:%M:%S.%fZ')
#End

#remove sessions of less than 2 interactions
train = removeShortSessions(train)


#delete records of items which appeared less than 5 times
itemLen = train.groupby('itemid').size() #groupby itemID and get size of each item
train = train[np.in1d(train.itemid, itemLen[itemLen > 4].index)]

#remove sessions of less than 2 interactions again
train = removeShortSessions(train)# 

'''
#Separate Data into Train and Test Splits
timeMax = data.Time.max() #maximum time in all records

------------------
sessionMaxTime = data.groupby('SessionID').Time.max() 
#group by sessionID and get the maximum time of each session LAST DAY OF EACH USER 
-------------------
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index #training split is all sessions that ended before the last day
sessionTest  = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index #testing split is all sessions has records in the last day

train = data[np.in1d(data.SessionID, sessionTrain)]
test = data[np.in1d(data.SessionID, sessionTest)]
'''
#Delete records in testing split where items are not in training split

#Convert To CSV
print('Full Training Set has', len(train), 'Events, ', train.visitorid.nunique(), 'visitors, and', train.itemid.nunique(), 'Items\n\n')
train.to_csv(dataAfter + 'recSys15TrainFull.csv',index=False)

#Separate Training set into Train and Validation Splits
timeMax = train.date.max()
print("timemaxxxxxxxxxxx" ,timeMax , type(timeMax))

sessionMaxTime = train.groupby('visitorid').date.max()
print("session time max",sessionMaxTime , type(sessionMaxTime))

#start:
# HERE WE TAKE THE LAST 2 DAYS AS Vlidation set 
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - datetime.timedelta(days=1))].index #training split is all sessions that ended before the last 2nd day
sessionValid = sessionMaxTime[sessionMaxTime >= (timeMax - datetime.timedelta(days=1))].index #validation split is all sessions that ended during the last 2nd day
trainTR = train[np.in1d(train.visitorid, sessionTrain)]
trainVD = train[np.in1d(train.visitorid, sessionValid)]
#End:

#Delete records in validation split where items are not in training split
trainVD = trainVD[np.in1d(trainVD.itemid, trainTR.itemid)]

#Delete Sessions in testing split which are less than 2
trainVD = removeShortSessions(trainVD)

#Convert To CSV
print('Training Set has', len(trainTR), 'Events, ', trainTR.visitorid.nunique(), 'Sessions, and', trainTR.itemid.nunique(), 'Items\n\n')
trainTR=trainTR.to_csv(dataAfter + 'recSys15TrainOnly.csv', index=False,  date_format='%Y-%m-%dT%H:%M:%S.%fZ')
print('Validation Set has', len(trainVD), 'Events, ', trainVD.visitorid.nunique(), 'Sessions, and', trainVD.itemid.nunique(), 'Items\n\n')
trainVD=trainVD.to_csv(dataAfter + 'recSys15Valid.csv', index=False , date_format ='%Y-%m-%dT%H:%M:%S.%fZ')