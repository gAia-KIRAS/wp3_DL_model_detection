import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


def postProcessingPixelLevelThresholding(masks_pred, threshold=0.9):
  """
    # apply thresholding
    # input:
        - masks_pred  (Bx128x128x2):
        - threshold  (default=0.5): smaller threshold -> more focus on landslide pixel, larger threshold -> strictly focus on landslide pixel,
    # output:
        - mask_list  (Bx128x128x2):
  """
  masks_pred = masks_pred.numpy()
  c1 = masks_pred[..., 0]
  c2 = np.where(masks_pred[..., 1] > threshold, 1, 0)  #channel 2 is landslide
  c1 = np.expand_dims(c1,axis=-1)
  c2 = np.expand_dims(c2,axis=-1)
  new_masks = np.concatenate((c1,c2), axis=-1)
  new_masks = tf.convert_to_tensor(new_masks, dtype=tf.float32)

  return new_masks


def postProcessingMorphology(masks_pred, op=1):
    """
        # apply morphology
        # input:
            - masks_pred  (Bx128x128x2):
        # output:
            - mask_list  (Bx128x128x2):
      """
    masks_pred = masks_pred.numpy()
    masks_pred = np.argmax(masks_pred, axis=-1) # Bx128x128
    masks_pred = masks_pred.astype(np.uint8)

    kernel = np.ones((4, 4), np.uint8)
    if op == 0: # opening - removing salt noise 
        masks_erosion  = cv2.erode(masks_pred, kernel, iterations=1)
        new_masks      = cv2.dilate(masks_erosion, kernel, iterations=1)
    else:       # closing - removing pepper noise
        masks_dilation = cv2.dilate(masks_pred, kernel, iterations=1)
        new_masks      = cv2.erode(masks_dilation, kernel, iterations=1)
  
    new_masks = tf.convert_to_tensor(new_masks, dtype=tf.uint8)
    new_masks = tf.one_hot(new_masks, depth=2, axis=-1)

    return tf.cast(new_masks, dtype=tf.float32)


def funcProcessingVotingMask(masks):
    masks = tf.image.resize(masks,[128,128],method=tf.image.ResizeMethod.BILINEAR)
    return np.asarray(masks)


def postProcessingVotingMask(masks_pred):
    """
        # apply voting
        # input:
            - masks_preds 3 x (Bx128x128x2):
        # output:
            - mask_list  (Bx128x128x2):
      """

    y_64  = masks_pred[2] 
    y_128 = masks_pred[1]
    y_256 = masks_pred[0]

    y_64   = funcProcessingVotingMask(y_64)
    y_128  = np.asarray(y_128)
    y_256  = funcProcessingVotingMask(y_256)

    new_masks = (y_64 + y_128 + y_256)/3.
    new_masks = tf.convert_to_tensor(new_masks)

    return tf.cast(new_masks, dtype=tf.float32)


def postProcessingMultiAngle(y,y90,y180,y270):
    """
        # apply multi angle
        # input:
            - masks_preds 4 x 3 x (Bx128x128x2):
        # output:
            - mask_list  (Bx128x128x2):
    """
    y90  = tf.image.rot90(y90,  k=3)
    y180 = tf.image.rot90(y180, k=2)
    y270 = tf.image.rot90(y270, k=1)

    new_masks = (y + y90 + y180 + y270) / 4.
    new_masks = tf.convert_to_tensor(new_masks)

    return tf.cast(new_masks, dtype=tf.float32)


def postProcessingAll(masks_pred):
  masks_pred = postProcessingPixelLevelThresholding(masks_pred, threshold=0.72)
  masks_pred = postProcessingMorphology(masks_pred, op=1)

  return masks_pred
