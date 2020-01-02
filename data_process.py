import os
import cv2 
import random
import numpy as npy
  

class DataIterEPF(object):
    def __init__(self, datadir,img_list, crop_num, crop_size, is_shuffle):
        self._datadir=datadir
        self._img_list=img_list
        self._crop_num=crop_num
        self._crop_size=crop_size
        self._is_shuffle=is_shuffle
        self._provide_input=zip(["img_in"],[(crop_num,3, crop_size, crop_size)])
        self._provide_output=zip(["img_out"],[(crop_num,3, crop_size, crop_size)])
        self._num_img=len(img_list)
        self._cur_idx=0
        self._iter_cnt=0
        
    def reset(self):
        self._cur_idx=0
        self._iter_cnt=0
        
    def fetch_next(self):
        if self._is_shuffle and npy.mod(self._cur_idx,self._num_img)==0:
            self._cur_idx=0
            random.shuffle(self._img_list) 
        crop_size=self._crop_size
        img_path1=os.path.join(self._datadir,self._img_list[self._cur_idx][0])
        img1=cv2.imread(img_path1, cv2.IMREAD_COLOR)
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img1 = img1.transpose(2, 0, 1)
        [nchl1, nrow1, ncol1 ]=img1.shape
        img_path2=os.path.join(self._datadir,self._img_list[self._cur_idx][1])
        img2=cv2.imread(img_path2, cv2.IMREAD_COLOR)
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        img2 = img2.transpose(2, 0, 1)
        [nchl, nrow, ncol]=img2.shape
        
        if (nrow1!=nrow) or ncol1!=ncol or nchl1 !=nchl:
            raise ValueError("Two images have different size")
     
        self._iter_cnt += 1
        self._cur_idx += 1
        if nrow < crop_size or ncol < crop_size:
            raise ValueError("Crop size is larger than image size")
        img1=img1.astype(npy.float32)
        img2=img2.astype(npy.float32)     
        sub_img1=npy.zeros((self._crop_num, 3, crop_size, crop_size))
        sub_img2=npy.zeros((self._crop_num, 3, crop_size, crop_size))
        for i in range(self._crop_num):
            nrow_start=npy.random.randint(0,nrow-crop_size)
            ncol_start=npy.random.randint(0,ncol-crop_size)
            img_crop=img1[:, nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size]              
            img_crop=img_crop/255.0       
            sub_img1[i,:,:,:]=img_crop
            
            img_crop=img2[:, nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size]
            img_crop=img_crop/255.0              
            sub_img2[i,:,:,:]=img_crop

        return (sub_img1.astype(npy.float32),sub_img2.astype(npy.float32))    
 

