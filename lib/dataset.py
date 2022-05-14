import pandas as pd
import numpy as np
import torch
import datetime
from datetime import datetime

class Dataset(object):
    def __init__(self, path, sep=',', session_key='visitorid', item_key='itemid', time_key='date', n_sample = -1 , itemmap = None, itemstamp = None, time_sort=False):
        # Read csv

        #start:
        df = pd.read_csv(path)
        df[time_key]=df[time_key].apply(lambda x:datetime. strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
        df[time_key]=df[time_key].apply(lambda x:x.timestamp())
        #End
        
        self.df = df
        # self.df = pd.read_csv(path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        if n_sample > 0:#if we want to train on sample but we will use all data
            self.df = self.df[:n_sample]

        # Add colummn item index to data
        
        self.add_item_indices(itemmap = itemmap)
        """
        Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        clicks within a session are next to each other, where the clicks within a session are time-ordered.
        """
        #SORT DATA BY VISITOR ID THEN FOR EACH USER ORDER IN TIME 
        self.df.sort_values([session_key, time_key], inplace=True)
        self.click_offsets = self.get_click_offset()
        self.session_idx_arr = self.order_session_idx()

    #HERE IT CREATE INDX COLUMN  ZERO BASED FOR EACH UNIQUE ITEM 
    #IF ITEM 5 HAS INDEX 100 ==> EACH TIME WILL TAKE THE SAME INDEX
    def add_item_indices(self, itemmap=None):
        """
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # type is numpy.ndarray
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            # Build itemmap is a DataFrame that have 2 columns (self.item_key, 'item_idx)
            itemmap = pd.DataFrame({self.item_key: item_ids,
                                   'item_idx': item2idx[item_ids].values})
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')


    #RETURN NUMPY ARRAY OF LENGTH NO. UNIQUE VISITORS + 1 THAT CONTAIN:
    #FIRST ELEMENT IS ZERO AND CUMMULATIVE SUM OF NO. OF ACTIONS IN EACH USER 
    #IF USER NO1 HAVE VIEW 5 ITEMS AND USER NO2 HAVE VIEW 10 TIMES  
    #offsets = 0 , 5 ,15
    def get_click_offset(self):
        """
        self.df[self.session_key] return a set of session_key
        self.df[self.session_key].nunique() return the size of session_key set (int)
        self.df.groupby(self.session_key).size() return the size of each session_id
        self.df.groupby(self.session_key).size().cumsum() retunn cumulative sum
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)#CREATE ZEROS NUMPY ARRAY OF size of session_key set (int)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    
    #HERE CREATE NUMPY ARRAY OF LENGTH OF UNIQUE VISITORS START FROM ZERO   
    def order_session_idx(self):
        if self.time_sort:
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            session_idx_arr = np.argsort(sessions_start_time)
    #HERE CREATE NUMPY ARRAY OF LENGTH OF UNIQUE VISITORS START FROM ZERO(THAT IS WHAT WE APPLY)
    # we will create session index start from zero  for each unique session  
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    #THAT WILL APPLY ADDING 1 COLUMN OF ITEM INDEX 
    @property
    def items(self):
        return self.itemmap[self.item_key].unique()

#class for creating session-parallel mini-batches.
class DataLoader():
    def __init__(self, dataset, batch_size = 50):
        """
        A class for creating session-parallel mini-batches.

        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.

        Yields:
            ##INPUT WILL BE VECTOR WITH LENGHT OF UNIQE ITEMS 
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            
            ##WILL BE VECTOR WITH LENGHT OF UNIQE ITEMS THAT STORES THE TARGET IREMS
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """
        # initializations

        df = self.dataset.df
        click_offsets = self.dataset.click_offsets#CREATE A CUMSUM ARRAY
        session_idx_arr = self.dataset.session_idx_arr#CREATE VISITOR INDX ARRAY 

        iters = np.arange(self.batch_size) # NP ARRAY OF SIZE 50 0==>49
        maxiter = iters.max() #49

        
        #session1 5 items , session2 10 items , session3 5 
        #for example click offset[0,5,15,20]
        # عشان انا عايز ال اندكس بتاع اول ايتم لكل يوزر و هيمشى ب الباتش 
        
        #[0,5,15,20]
        #IF BATCH = 3  
        #START = [0,5,15] START OF EACH USER IN BATCH 
        #END = [5,15,20] END OF EACH USER IN BATCH

        #START IS THE STRART OF BATCH SESSIONS 
        #END IS THE END OF X SESSIONS

        start = click_offsets[session_idx_arr[iters]]#start of each visitor 
        end =   click_offsets[session_idx_arr[iters] + 1] #end of each visitor
        mask = []  # indicator for the sessions to be terminated
        finished = False


        
        while not finished:
        
            minlen = (end - start).min()#اقل سيشن فيها ايتمزعشان لما يخلصها يشيلها و يحط واحده تانيه مكانها 
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]#هنا بيجيب ال ايتيمز الموجوده ف ال ستارت 

            for i in range(minlen - 1):#عشان يمشى على ايتيم الى قبل الاخير و يبقى الاخير هو التارجت
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask 

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1) #عشان السيشنز الكبيره الى عايز اكمل فيها هيبقى الاستارت هو الى نهاية السيشن الصفيره 
            # see if how many sessions should terminate
            #ex : session1 = 5 , se2 = 10 , se3 = 5
            #min(len) = 5
            #start = [0,5,15]
            #end = [5,15,20]
            #بعد ما اصغر سيشن تخلص 
            #start = [4,9,19] will be
            #end = [5,15,20]
            #mask = (0,1,2)[5-4 == 1 True , 15-9=6 False , 20-19=1 true ]==> out is [0 , 2] ده الماسك بتاعى


            #عشان تعرف ال سيشنز الى خلصت 
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1 #max =  50
                if maxiter >= len(click_offsets) - 1:#click_offsets = UNIQUE_USERS+1 IF MAXIRER >= UNIQUE USERS ALL USERS ARE FINISH  
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter #start from 50 هتبقى مكان الى خلص حط لى عليه الدور 
                start[idx] = click_offsets[session_idx_arr[maxiter]] # start will be 50
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]
