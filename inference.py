import tensorflow as tf
import numpy as np
import rasterio as rs
import os
import math
import cv2
import pickle
from PIL import Image
from datetime import datetime
from losses import *
import csv
from post_processing import *
from hyper_parameter import *
import datetime;

def readTifFile(tif_file_head):
    c = 0
    
    ## read with rasterio
    tif_file_tail = '.tif'

    for i in ['02','03','04','08','11']:
        tif_file_name = tif_file_head + i + tif_file_tail
        tif_file = rs.open(tif_file_name)
    
        a = np.array(tif_file.read()) # (1,10980,10980)
        a = np.where( np.isnan(a)==True, 0 , a) # black pixel for Null 
        a = a[0]

        ## resize band 11 because it was collected as 20m resolution
        if i == "11":
            a = cv2.resize(a, dsize=(10980, 10980), interpolation=cv2.INTER_LINEAR)

        a = np.expand_dims(a, axis=-1) # (10980,10980,1)

        if c == 0:
            res = a
        else:
            res = np.concatenate((res,a),axis=-1)

        c = c + 1

    return res  # (10980, 10980, 5)


def getTotalNumImgs(width=10980, height=10980):
    num_w = math.ceil(width / 128)
    num_h = math.ceil(height / 128)
    
    return num_w * num_h


def getTotalBatchNum(size=10980, train_size=128):
    if size % train_size == 0:
        return (int(size/train_size), 0)
    return (int(size/train_size + 1), size % train_size)


def getOnePiece(l_img, batch_w, batch_h):
    """
        - input:
            a large WxH image of a band data: 3d numpy array
        - output:
            a 128x128 image of 1 band data: 3d numpy array
    """
    
    s_img = l_img[batch_w:batch_w+128 , batch_h:batch_h+128, :]

    return s_img


def applyFeatureEngineering(x):
    """
        - input: 
            x - 128x128x5 (b,g,r,b8,b11) 
        - ouput:
            x - 128x128x13
    """
    ## rgb normalization
    red   = (x[:,:,2]-np.min(x[:,:,2])) / (np.max(x[:,:,2]) - np.min(x[:,:,2]) + 1e-8)
    green = (x[:,:,1]-np.min(x[:,:,1])) / (np.max(x[:,:,1]) - np.min(x[:,:,1]) + 1e-8)
    blue  = (x[:,:,0]-np.min(x[:,:,0])) / (np.max(x[:,:,0]) - np.min(x[:,:,0]) + 1e-8)

    red   = np.expand_dims(red, axis=2)
    green = np.expand_dims(green, axis=2)
    blue  = np.expand_dims(blue, axis=2)

    x     = np.concatenate((x, red, green, blue), axis=-1)
    del red, green, blue

    ## ndvi
    red   = x[:,:,2] # B4
    nir   = x[:,:,3] # B8
    nvdi  = (nir-red) / np.clip(nir+red, a_min=1e-8, a_max=None)
    nvdi  = np.expand_dims(nvdi, axis=2)
    
    x     = np.concatenate((x, nvdi), axis=-1)
    del red, nir, nvdi

    ## vi
    b8   = x[:,:,3]  # B8
    b11  = x[:,:,4]  # B11
    vi   = (b8-b11) / np.clip(b8+b11, a_min=1e-8, a_max=None)
    vi   = np.expand_dims(vi, axis=2)

    x    = np.concatenate((x, vi), axis=-1)
    del b8,b11, vi

    ## gray
    gray = (x[:,:,0] + x[:,:,1] + x[:,:,2]) / 3
    gray = np.expand_dims(gray, axis=2)

    x    = np.concatenate((x, gray), axis=-1)
    del gray

    ## blur
    gray  = x[:,:,10] #
    gray  = (gray-np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8)
    gray *= 255.
    gray  = gray.astype(np.uint8)

    blur  = cv2.blur(gray,(10,10))
    blur  = blur.astype(np.float32)
    blur /= 255.
    blur  = np.expand_dims(blur, axis=2)
    x     = np.concatenate((x, blur), axis=-1)

    blur  = cv2.medianBlur(gray,15)
    blur  = blur.astype(np.float32)
    blur /= 255.
    blur  = np.expand_dims(blur, axis=2)
    x     = np.concatenate((x, blur), axis=-1)
    del gray, blur

    return x


def predictLandslide(model, x_seq, img_index, batch_test, info_time, img_per_rowcol=86):

    #strategy = tf.distribute.MirroredStrategy()
    y_pred = model(x_seq)  

    # take the 128x128 resolution prediction only
    y_pred = y_pred[1]      # Bx128x128x2
    real_mask = y_pred[:,:,:,1]   # added by LP to collect the predicted prob
    
    ## apply thresholding or not ?
    y_pred = postProcessingPixelLevelThresholding(y_pred, 0.95)

    ## find pixel's class of y_pred
    mask = tf.math.argmax(y_pred, axis=-1) # Bx128x128
    
    ## check which small image has landslide (threshold: small image has greater than 10 landslide pixels)
    y_pred = tf.math.reduce_sum(mask, axis=[1,2]) # (B,)
    
    ## image has less than 10 predicted landslide pixel will be mark as 0 else 1
    tmp    = tf.where(y_pred < 10, 0, 1 ).numpy()  # y_pred < 10

    re_list = []
    ## loop through every images in a batch
    for i in range(tmp.shape[0]):
        if tmp[i] == 1:
            if img_index%batch_test == 0:
                row = (img_index - batch_test + i) // img_per_rowcol 
                col = (img_index - batch_test + i) - row*img_per_rowcol       
            else: #for final batch: 46
                row = (img_index - (img_index%batch_test) + i) // img_per_rowcol 
                col = (img_index - (img_index%batch_test) + i) - row*img_per_rowcol      
            #with open("./debug_log.txt", "a") as text_file:
            #    text_file.write("DEBUG: col :{}, row:{} \n".format(col, row))

            #re_list.append((row, col, mask[i].numpy()))         
            re_list.append((row, col, mask[i].numpy(), real_mask[i].numpy(), info_time))         
            if row > img_per_rowcol or col > img_per_rowcol or mask[i].shape != (128,128):
                print("Something wrong due to the row col assignment !")
                exit()
    return re_list


def predictOneImage(model, l_img, info_time):
    ## find number of small images (batch_w*batch_h)
    batch_w = getTotalBatchNum()   # (b,n_last_batch)
    batch_h = getTotalBatchNum()   # (b,n_last_batch) 

    batch_test =  30

    count = 0 #  to count number of images currently in a batch
    img_index = 0 
    re_list = []
    for h in range(batch_h[0]):
        ## handle last batch
        if h == batch_h[0]-1 and batch_h[1] != 0:
           h_ind = 10980 - 128 - 1
        else:
           h_ind = h*128 

        for w in range(batch_w[0]):
            ## handle last batch
            if w == batch_w[0]-1 and batch_w[1] != 0:
                w_ind = 10980 - 128 - 1
            else:
                w_ind = w*128


            ## get batch of pieces -> transform
            #x_infer = getOnePiece(l_img, w, h)                 # 128x128x5
            x_infer = getOnePiece(l_img, w_ind, h_ind)          # 128x128x5   #LP

            x_infer = applyFeatureEngineering(x_infer)          # 128x128x13
            x_infer = np.expand_dims(x_infer, axis=0)           # 1x128x128x13

            ## check if the shape of x is valid
            if x_infer.shape != (1,128,128,13):
                print('Some thing wrong ? ')
                return

            ## stack to form a batch
            if count == 0:
                x_seq = x_infer                                  # Bx128x128x5
            else:
                x_seq = np.concatenate((x_seq, x_infer), axis=0) # Bx128x128x5

            ## increase number of images in a batch
            count += 1
            img_index += 1

            ## make predictions
            if (count == batch_test) or ((h == batch_h[0]-1) and (w == batch_w[0]-1)): # count == 30 for 2080 and count == 50 for TITAN
                re_list.extend(predictLandslide(model, x_seq, img_index, batch_test, info_time))
                count = 0
    
    return re_list


def savePrediction(file_name, pred_list, save_opt='pkl'):
    if save_opt == 'pkl':
        file_tail = '.pkl'
    elif save_opt == 'txt':
        file_tail = '.txt'
    else:
        return 

    ## check if file name exist or not   
    file_name = './11_output_pkl/' +  file_name + file_tail
    if os.path.exists(file_name):
        print('Duplicated file name ?')
        exit()

    ls_flag = 'True' if len(pred_list) != 0 else 'False'
    
    ## save as .pkl file
    with open(file_name, 'wb') as output:
        pickle.dump(pred_list, output)

    ## save as .txt file
    #with open(file_name, "w") as output:
        #output.write("Image name: {}  ; Have landslide or not ?: {} \n\n Prediction list (tuples of start row and start colummn): \n {}".format(file_name, ls_flag,str(pred_list)))
    
    return


def doInference():
    """
        - output: 
            a batch of 128x128 pieces
    """
    ## load model
    model_path = './01_pre_trained_models/model.h5'
    model = tf.keras.models.load_model( filepath=model_path,
                                        custom_objects={ 'FocalLoss': FocalLoss, 'IOULoss': IOULoss})
    
    ## load large image
    img_dir = './02_input_images/'
    file_names = os.listdir(img_dir)

    ## remove file if it not contain .tif at the end and take out the unique name of each image
    file_names = [img for img in file_names if len(img.split('.'))==2]           # take .tif files only, remove .xml, .aux files
    file_names = [img for img in file_names if img.split('L2A_')[1]=='SCL.tif']  # take the unique name of an image (each image has only 1 SCL file)(instead *L2A_B2, *L2A_B3,.., *L2A_SCL => *L2A_)
    file_names = [img.split('SCL')[0] for img in file_names]       # remove SCL.tif

    ## predict large image by large image
    for file_name in file_names:
        tif_file_head = img_dir + file_name + 'B' # B will be concatenate with 02,03,04,08,11

        ## read large image with rasterio   #concat 5 band toghether 
        l_img = readTifFile(tif_file_head)

        # stores current time to predict
        ct = str(datetime.datetime.now())
        yy = ct.split('-')[0]
        mm = ct.split('-')[1]
        dd = ct.split('-')[2].split(' ')[0]

        hh = ct.split(' ')[-1].split(':')[0]
        mi = ct.split(' ')[-1].split(':')[1]
        ss = ct.split(' ')[-1].split(':')[2].split('.')[0]
        info_time = yy+mm+dd+'_'+hh+mi+ss

        # predict list
        pred_list = predictOneImage(model, l_img, info_time)

        ## save prediction
        savePrediction(file_name, pred_list)
                            
    return 


if __name__ == '__main__':
    doInference()
