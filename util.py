import numpy as np
#import matplotlib.pyplot as plt
import math
import os
import argparse
import cv2
from natsort import natsorted
import h5py

from hyper_parameter import *

def visualizeOneImg(img, op=0, is_rgb=False):
  if not isinstance(img, np.ndarray):
    img = img.numpy()

  if is_rgb:
    rgb = img[:,:,0:3]
  else:
    rgb = img[:,:,1:4]

  if op == 0:
    rgb = changeBGR2RGB(rgb)
  elif op == 1:
    rgb = changeBGR2RGB_2(rgb)
  else:
    rgb = rgb

  plt.imshow(rgb)
  plt.title('RGB Image')
  plt.show()

  return


def visualizeOneMask(mask):
  if not isinstance(mask, np.ndarray):
    mask = mask.numpy()

  plt.imshow(mask, cmap='gray')
  plt.title('Mask Image')
  plt.colorbar()
  plt.show()
  
  return


def changeBGR2RGB(bgr_img):
  """
    convert BGR to RGB image using Min-Max scaling
  """
  if not isinstance(bgr_img, np.ndarray):
    bgr_img = bgr_img.numpy()

  red   = (bgr_img[:,:,2]-np.min(bgr_img[:,:,2])) / (np.max(bgr_img[:,:,2])-np.min(bgr_img[:,:,2]))
  green = (bgr_img[:,:,1]-np.min(bgr_img[:,:,1])) / (np.max(bgr_img[:,:,1])-np.min(bgr_img[:,:,1]))
  blue  = (bgr_img[:,:,0]-np.min(bgr_img[:,:,0])) / (np.max(bgr_img[:,:,0])-np.min(bgr_img[:,:,0]))

  red   = np.expand_dims(red, axis=2)
  green = np.expand_dims(green, axis=2)
  blue  = np.expand_dims(blue, axis=2)

  rgb   = np.concatenate((red, green), axis=-1)
  rgb   = np.concatenate((rgb, blue), axis=-1)

  return rgb


def changeBGR2RGB_2(bgr_img):
  """
    convert BGR to RGB image
  """
  if not isinstance(bgr_img, np.ndarray):
    bgr_img = bgr_img.numpy()

  max_pixel_value = 3000# 10000 # tuning parameter

  red   = np.clip((bgr_img[:,:,2])*HyperParameter().img_mean[3] / max_pixel_value, 0, 1) # HyperParameter().img_max[3]
  green = np.clip((bgr_img[:,:,1])*HyperParameter().img_mean[2] / max_pixel_value, 0, 1) # HyperParameter().img_max[2]
  blue  = np.clip((bgr_img[:,:,0])*HyperParameter().img_mean[1] / max_pixel_value, 0, 1) # HyperParameter().img_max[1]

  red   = np.expand_dims(red, axis=2)
  green = np.expand_dims(green, axis=2)
  blue  = np.expand_dims(blue, axis=2)

  rgb   = np.concatenate((red, green), axis=-1)
  rgb   = np.concatenate((rgb, blue), axis=-1)

  return rgb


def addRGB(multispectral_img, is_rgb=False):
  """
    add RGB channels to image using Min-Max scaling
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  if is_rgb:
    bgr = multispectral_img[:,:,0:3] # BGR
  else:
    bgr = multispectral_img[:,:,1:4] # BGR

  red   = (bgr[:,:,2]-np.min(bgr[:,:,2])) / (np.max(bgr[:,:,2])-np.min(bgr[:,:,2]))
  green = (bgr[:,:,1]-np.min(bgr[:,:,1])) / (np.max(bgr[:,:,1])-np.min(bgr[:,:,1]))
  blue  = (bgr[:,:,0]-np.min(bgr[:,:,0])) / (np.max(bgr[:,:,0])-np.min(bgr[:,:,0]))

  red   = np.expand_dims(red, axis=2)
  green = np.expand_dims(green, axis=2)
  blue  = np.expand_dims(blue, axis=2)

  multispectral_img = np.concatenate((multispectral_img, red), axis=-1)
  multispectral_img = np.concatenate((multispectral_img, green), axis=-1)
  multispectral_img = np.concatenate((multispectral_img, blue), axis=-1)

  return multispectral_img


def addRGB_2(multispectral_img, is_rgb=False):
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  if is_rgb:
    bgr = multispectral_img[:,:,0:3] # BGR
  else:
    bgr = multispectral_img[:,:,1:4] # BGR

  red   = np.clip((bgr[:,:,2]*HyperParameter().img_mean[3] ) /  6000, 0, 1)# HyperParameter().img_max[3]
  green = np.clip((bgr[:,:,1]*HyperParameter().img_mean[2] ) /  6000, 0 , 1) # HyperParameter().img_max[2]
  blue  = np.clip((bgr[:,:,0]*HyperParameter().img_mean[1] ) /  6000, 0, 1) # HyperParameter().img_max[1]

  red   = np.expand_dims(red, axis=2)
  green = np.expand_dims(green, axis=2)
  blue  = np.expand_dims(blue, axis=2)

  multispectral_img = np.concatenate((multispectral_img, red), axis=-1)
  multispectral_img = np.concatenate((multispectral_img, green), axis=-1)
  multispectral_img = np.concatenate((multispectral_img, blue), axis=-1)

  return multispectral_img


def addNDVI(multispectral_img):
  """
    NDVI = (NIR-RED) / (NIR+RED) = (B8-B4)/(B8+B4)
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  red   = multispectral_img[:,:,3] # B4
  nir   = multispectral_img[:,:,7] # B8
  nvdi  = (nir-red) / np.clip(nir+red, a_min=1e-8, a_max=None)
  nvdi  = np.expand_dims(nvdi, axis=2)

  multispectral_img = np.concatenate((multispectral_img, nvdi), axis=-1)

  return multispectral_img

def addVegetationIndex(multispectral_img):
  """
    VI = (B8-B11)/(B8+B11)
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  b8   = multispectral_img[:,:,7]  # B8
  b11  = multispectral_img[:,:,10] # B11
  vi   = (b8-b11) / np.clip(b8+b11, a_min=1e-8, a_max=None)
  vi   = np.expand_dims(vi, axis=2)

  multispectral_img = np.concatenate((multispectral_img, vi), axis=-1)

  return multispectral_img


def addNBR(multispectral_img):
  """
    NBR = (NIR-SWIR) / (NIR+SWIR) = (B8-B12)/(B8+B12)
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
        multispectral_img = multispectral_img.numpy()

  swir  = multispectral_img[:,:,11] # B12
  nir   = multispectral_img[:,:,7] # B8
  nbr   = (nir-swir) / np.clip(nir+swir, a_min=1e-8, a_max=None)
  nbr   = np.expand_dims(nbr, axis=2)

  multispectral_img = np.concatenate((multispectral_img, nbr), axis=-1)

  return multispectral_img


def addGray(multispectral_img):
  """
    gray = (B2+B3+B4)/3
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  b    = multispectral_img[:,:,1] # B2
  g    = multispectral_img[:,:,2] # B3
  r    = multispectral_img[:,:,3] # B4
  gray = (b + g + r) / 3
  gray = np.expand_dims(gray, axis=2)

  multispectral_img = np.concatenate((multispectral_img, gray), axis=-1)

  return multispectral_img


def addEdge(multispectral_img):
  """
    Canny Edge Detection
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  gray  = multispectral_img[:,:,20] #
  gray  = (gray-np.min(gray)) / (np.max(gray)-np.min(gray))
  gray *= 255
  gray  = gray.astype(np.uint8)

  edge  = cv2.Canny(gray,150,227)
  edge  = edge.astype(np.float32)
  edge /= 255.
  edge = np.expand_dims(edge, axis=2)

  multispectral_img = np.concatenate((multispectral_img, edge), axis=-1)

  return multispectral_img


def addBlur(multispectral_img):
  """
    Gaussian and Median Blurring
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  gray  = multispectral_img[:,:,20] #
  gray  = (gray-np.min(gray)) / (np.max(gray)-np.min(gray))
  gray *= 255.
  gray  = gray.astype(np.uint8)

  blur  = cv2.blur(gray,(10,10))
  blur  = blur.astype(np.float32)
  blur /= 255.
  blur  = np.expand_dims(blur, axis=2)
  multispectral_img = np.concatenate((multispectral_img, blur), axis=-1)

  blur  = cv2.medianBlur(gray,15)
  blur  = blur.astype(np.float32)
  blur /= 255.
  blur  = np.expand_dims(blur, axis=2)
  multispectral_img = np.concatenate((multispectral_img, blur), axis=-1)

  return multispectral_img


def addGradient(multispectral_img):
  """
    Gradient along x-axis and y-axis
  """
  if not isinstance(multispectral_img, np.ndarray): # convert tf.tensor to np.array
    multispectral_img = multispectral_img.numpy()

  norm_factor = 27.32001
  gray  = multispectral_img[:,:,20] #
  gray  = (gray-np.min(gray)) / (np.max(gray)-np.min(gray))

  sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
  sobel_x  = sobel_x.astype(np.float32)
  sobel_x /= norm_factor
  sobel_x  = np.expand_dims(sobel_x, axis=2)
  multispectral_img = np.concatenate((multispectral_img, sobel_x), axis=-1)

  sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
  sobel_y  = sobel_y.astype(np.float32)
  sobel_y /= norm_factor
  sobel_y  = np.expand_dims(sobel_y, axis=2)
  multispectral_img = np.concatenate((multispectral_img, sobel_y), axis=-1)

  return multispectral_img


def findLandslideImage(data_dir, test_fold=1):
    """
        # find the list of masks in training set that consist of land slide area
        # input:
            - data_dir: training directory
            - split_ratio: train ratio
        # output:
            - a list of masks with landsliding
            - a list of relative images
    """
  
    images_dir  = data_dir + '/img/'
    masks_dir   = data_dir + '/mask/'

    n_img = len(os.listdir(masks_dir))
    #n_img_train = math.ceil( len(os.listdir(masks_dir)) * split_ratio )
    #print("number of training images: ",n_img_train)

    mask_list   = os.listdir(masks_dir)
    mask_list   = natsorted(mask_list)
    image_list  = os.listdir(images_dir)
    image_list  = natsorted(image_list)
    

    if test_fold == 1:
        mask_list   = mask_list[:int(n_img*0.8)] + mask_list[int(n_img*1):]
        image_list  = image_list[:int(n_img*0.8)] + image_list[int(n_img*1):]
    elif test_fold == 2:
        mask_list   = mask_list[:int(n_img*0.6)] + mask_list[int(n_img*0.8):]
        image_list  = image_list[:int(n_img*0.6)] + image_list[int(n_img*0.8):]
    elif test_fold == 3:
        mask_list   = mask_list[:int(n_img*0.4)] + mask_list[int(n_img*0.6):]
        image_list  = image_list[:int(n_img*0.4)] + image_list[int(n_img*0.6):]
    elif test_fold == 4:
        mask_list   = mask_list[:int(n_img*0.2)] + mask_list[int(n_img*0.4):]
        image_list  = image_list[:int(n_img*0.2)] + image_list[int(n_img*0.4):]
    elif test_fold == 5:
        mask_list   = mask_list[:int(n_img*0.0)] + mask_list[int(n_img*0.2):]
        image_list  = image_list[:int(n_img*0.0)] + image_list[int(n_img*0.2):]

    n_img_train = len(mask_list)
    print("number of training images: ",n_img_train)

    landslide_mask  = []
    landslide_image = []

    for i in range(n_img_train):
        image_name = image_list[i]
        mask_name  = mask_list[i]
        mask_open  = os.path.join(masks_dir, mask_name)

        ## load mask
        f_mask     = h5py.File(mask_open, 'r')
        one_mask   = f_mask[list(f_mask.keys())[0]]
        one_mask   = np.asarray(one_mask) # (128,128)
        f_mask.close()

        if (np.sum(one_mask)) != 0:
            landslide_mask.append(mask_name)
            landslide_image.append(image_name)

    return landslide_mask, landslide_image


def init_argparse():
  parser = argparse.ArgumentParser(
    usage="%(prog)s --split_ratio XXX",
    description="Set split ratio" 
  )
  parser.add_argument("--split_ratio", required=True, help=' --split_ratio 0.8')
    
  return parser



if __name__ == '__main__':
  parser = init_argparse()
  args   = parser.parse_args()
  split_ratio = float(args.split_ratio)

  landslide_mask, landslide_image = findLandslideImage('../dataset/train', split_ratio)
  print(len(landslide_mask))
  print(len(landslide_image))
  print(landslide_mask[:10])
  print(landslide_image[:10])
  print("ratio of landslide images: ", len(landslide_image) / math.ceil(len(os.listdir('../dataset/train/img'))*0.8))

  landslide_mask, landslide_image = findLandslideImage('../dataset/train', 1)
  print(len(landslide_mask))
  print(len(landslide_image))
  print(landslide_mask[:10])
  print(landslide_image[:10])
  print("ratio of landslide images: ", len(landslide_image) / math.ceil(len(os.listdir('../dataset/train/img'))*1))

  del landslide_mask, landslide_image
