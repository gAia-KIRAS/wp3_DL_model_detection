import numpy as np
from natsort import natsorted
import h5py
import os

from hyper_parameter import *

def checkPixelValueRange(data_dir, op):
    
    l = os.listdir(data_dir)
    l = natsorted(l)
    
    max_l = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    min_l = [99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999,99999]

    n_img = 0
    for i in range(len(l)):
        img_open = data_dir + l[i]

        ## load mask
        f_img = h5py.File(img_open, 'r')
        img   = f_img[list(f_img.keys())[0]]
        img   = np.asarray(img) # (128,128)
        f_img.close()
        n_img+= 1

        for j in range(len(max_l)):
            tmax = np.max(img[:,:,j])
            tmin = np.min(img[:,:,j])
            if tmax > max_l[j]:
                max_l[j] = tmax
            if tmin < min_l[j]:
                min_l[j] = tmin

    print("===== ",op,"SET =====")
    print("n_imgs: ",n_img,'\n')
    print("B1 - coastal aerosol: ", max_l[0],min_l[0])
    print("B2 - blue: ", max_l[1],min_l[1])
    print("B3 - green: ", max_l[2],min_l[2])
    print("B4 - red: ", max_l[3],min_l[3])
    print("B5 - vre: ", max_l[4],min_l[4])
    print("B6 - vre: ", max_l[5],min_l[5])
    print("B7 - vre: ", max_l[6],min_l[6])
    print("B8 - nir: ", max_l[7],min_l[7])
    print("B9 - wv: ", max_l[8],min_l[8])
    print("B10 - swir - cirrus: ", max_l[9],min_l[9])
    print("B11 - swir: ", max_l[10],min_l[10])
    print("B12 - swir: ", max_l[11],min_l[11])
    print("B13 - slope: ", max_l[12],min_l[12])
    print("B14 - dem: ", max_l[13],min_l[13])
    print(np.array(max_l)*np.array(HyperParameter().img_mean))

    return


def landslideStatistic(data_dir):
    images_dir  = data_dir + '/img/'
    masks_dir   = data_dir + '/mask/'

    n_img_train = len(os.listdir(masks_dir)) 
    print("number of training images: ",n_img_train)

    mask_list   = os.listdir(masks_dir)
    mask_list   = natsorted(mask_list)

    landslide_image = 0
    landslide_percent = 0
    for i in range(n_img_train):
        mask_name  = mask_list[i]
        mask_open  = os.path.join(masks_dir, mask_name)

        # load mask
        f_mask     = h5py.File(mask_open, 'r')
        one_mask   = f_mask[list(f_mask.keys())[0]]
        one_mask   = np.asarray(one_mask) # (128,128)
        f_mask.close()

        if (np.sum(one_mask)) != 0:
            landslide_image += 1
            landslide_percent += np.sum(one_mask)

    print("number of images contatain landslide and percentage: ", landslide_image, " -- ", landslide_image / n_img_train * 100, " %")
    print("percentage of sliding region: ", landslide_percent / (128*128*n_img_train) * 100, ' %')

    return 

if __name__ == '__main__':
    checkPixelValueRange(data_dir='../dataset/train/img/', op="TRAIN")
    print('\n\n')
    checkPixelValueRange(data_dir='../dataset/val/img/', op="VAL")
    print('\n\n')
    checkPixelValueRange(data_dir='../dataset/test/img/', op="TEST")
    print('\n\n')
    landslideStatistic("../dataset/train")
