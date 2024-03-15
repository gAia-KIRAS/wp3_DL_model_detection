import tensorflow as tf
import numpy as np 
import os
import platform

from dataset import *
from data_generator import *
from model import *
from train import *
from test import *
from util import *


def pred_vis():
    prepareDataset(data_dir='../zip_data') # 

    stored_dir      = '../results/20-mono-f-i-23-RANet-multi-best/' 
    best_model_file = stored_dir + 'model.h5'
    BATCH_SIZE      = 1   
    is_multi_resolution = 1

    ## initialize data generator
    generator = DataGenerator(data_dir='../dataset/train', batch_size=BATCH_SIZE, train_ratio=0.8, band_opt=1, is_multi_resolution=is_multi_resolution) # band_opt: 0->14, 1->23, 2->9

    ## load model
    model = loadModel(best_model_file)  

    ## run Data_Generator
    x_test, y_true_test_256, y_true_test_128, y_true_test_64, n_imgs = generator.getBatch(batch_num=27, is_aug=False, is_train=False, is_cutmix=False)

    ## make predictions
    y_pred_test = model(x_test)

    ## do post processing
    y_pred_test[1] = doPostProcessing(x_test, y_pred_test, model, 1)

    print(x_test.shape, x_test[0].shape)
    print(y_true_test_128.shape, y_true_test_128[0].shape)
    print(y_pred_test[1].shape, y_pred_test[1][0].shape)

    ## do visualization
    # visualizeOneImg(x_test[0], op=0)      
    # visualizeOneMask(y_true_test_128[0]) 
    # visualizeOneMask(tf.math.argmax(y_pred_test[1],axis=-1)[0]) 

    del model, generator, y_pred_test, x_test, y_true_test_64, y_true_test_128, y_true_test_256, n_imgs
    
    return


if __name__ == '__main__':
    pred_vis()
