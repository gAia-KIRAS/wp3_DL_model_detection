
import rasterio as rs
import numpy as np
import h5py
from PIL import Image
from hyper_parameter import *


def checkTifFile():
    ## read with rasterio
    #tif_file = rs.open('/var/data/storage/datasets/image/01_AI_for_Earth/02_slandslide/04_data_gaia_S2/2018/33TUM/NDVI_raw/33_T_UM_2018_10_S2A_33TUM_20181013_0_L2A_NDVI.tif')
    #print(tif_file.name)
    #print(tif_file.count)
    #print(tif_file.indexes)
    #print('\n\n Shape of image if open .tif file with rasterio: ',tif_file.width, tif_file.height)
    #a = np.array(tif_file.read())

    ## read with Pillow
    #tif = np.asarray(Image.open("/var/data/storage/datasets/image/01_AI_for_Earth/02_slandslide/04_data_gaia_S2/2018/33TUM/NDVI_raw/33_T_UM_2018_10_S2A_33TUM_20181013_0_L2A_NDVI.tif"))


    ## load h5py file
    h5py_file  = '../dataset/train/img/image_1.h5' 
    h5py_file  = h5py.File(h5py_file, 'r')
    h5py_file  = h5py_file[list(h5py_file.keys())[0]]
    h5py_arr   = np.asarray(h5py_file) # (128,128,14)
    img_mean   = np.reshape(np.array(HyperParameter().img_mean), (1, 1, -1))
    
    h5py_arr   = np.multiply(h5py_arr, img_mean)    
    
    for i in [2,3,4,8,11]:
        print('band ', i, ': ',HyperParameter().img_mean[i-1])
            

    ## check 5 bands: B2,B3,B4,B8,B11
    tif_folder = '/var/data/storage/datasets/image/01_AI_for_Earth/02_slandslide/04_data_gaia_S2/2018/33TUM/tmp/33_T_UM_2018_10_S2A_33TUM_20181013_0_L2A_B'
    for i in ['02','03','04','08','11']:
        tif_open  = tif_folder + str(i) + '.tif'
        tif_file  = rs.open(tif_open)
        tif_file  = np.array(tif_file.read())

        print('\n\n\n ================================= ') 
        print('TIF file band ', i, ': ' , np.max(tif_file), np.min(tif_file), np.mean(tif_file), np.std(tif_file))
        print('h5py file band ', i, ': ', np.max(h5py_arr[:,:,int(i)-1]), np.min(h5py_arr[:,:,int(i)-1]), np.mean(h5py_arr[:,:,int(i)-1]), np.std(h5py_arr[:,:,int(i)-1]))
    

    
    return


if __name__ == '__main__':
    checkTifFile()
