import tensorflow as tf

#####################################################################################
#####   All of metrics were coded to be optimized for gpu parallel computation   ####
#####################################################################################


def Accuracy(y_true, y_pred):
    """
        # calculate on batch
        # return: total correct pixels of a batch of images
    """

    ## find class of a pixel of y_pred
    y_pred = tf.math.argmax(y_pred, axis=-1)

    ## shoud be: (B,W,H)
    if y_true.shape != y_pred.shape:
        print('Error metric !')
        raise ValueError('Something error in Accuracy calculation')

    y_true   = tf.cast(y_true, dtype=tf.float32)
    y_pred   = tf.cast(y_pred, dtype=tf.float32)
    correct  = tf.cast(y_true == y_pred, dtype=tf.float32)
    accuracy = tf.reduce_sum(correct)
    accuracy = accuracy.numpy()

    return accuracy


def calculateF1(AandB, AorB):
    """
        # calculate on epoch (all batches)
        # return: mean f1 score
    """
    f1           = tf.where(tf.equal(AorB, 0), 1, 2*AandB / AorB)
    mean_f1      = tf.reduce_mean(f1)

    return tf.cast(mean_f1, dtype=tf.float32).numpy()


def F1(y_true, y_pred): # F1/2 < IOU < F1
    """
        # calculate on batch
        # return:
            - AndB
            - AorB
    """

    ## find pixel's class of y_pred
    y_pred = tf.math.argmax(y_pred, axis=-1)

    ## shoud be: (B,W,H)
    if y_true.shape != y_pred.shape:
        print('Error metric !')
        raise ValueError('Something error in Accuracy calculation')
  
    b,h,w    = y_true.shape
    ## flatten
    y_pred = tf.reshape(y_pred,[-1]) # (B*W*H,)
    y_true = tf.reshape(y_true,[-1]) # (B*W*H,)

    ## cast type
    y_true = tf.cast(y_true, dtype=tf.uint8)
    y_pred = tf.cast(y_pred, dtype=tf.uint8)

    ## one-hot
    y_true = tf.one_hot(y_true, depth=2, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=2, axis=-1)

    ## calculate f1
    AandB  = y_true * y_pred        # |A.B|
    AorB   = y_true + y_pred        # |A|+|B|

    AandB  = tf.reduce_sum(AandB, axis=0) # (c,) array
    AorB   = tf.reduce_sum(AorB, axis=0)  # (c,) array

    return AandB, AorB


def calculateMIOU(intersection, union):
    """
        # calculate on epoch (all batches)
        # return: mean IOU
    """
    iou      = tf.where(tf.equal(union, 0), 1, intersection / union)
    mean_iou = tf.reduce_mean(iou)

    return tf.cast(mean_iou, dtype=tf.float32).numpy()


def MIOU(y_true, y_pred):
    """
        # calculate on batch
        # return:
            - intersection
            - union
    """

    ## find pixel's class of y_pred
    y_pred = tf.math.argmax(y_pred, axis=-1)

    ## shoud be: (B,W,H)
    if y_true.shape != y_pred.shape:
        print('Error metric !')
        raise ValueError('Something error in Accuracy calculation')

    ## flatten
    y_pred = tf.reshape(y_pred,[-1]) # (B*W*H,)
    y_true = tf.reshape(y_true,[-1]) # (B*W*H,)

    # cast type
    y_true = tf.cast(y_true, dtype=tf.uint8)
    y_pred = tf.cast(y_pred, dtype=tf.uint8)

    # one-hot
    y_true = tf.one_hot(y_true, depth=2, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=2, axis=-1)

    ## calculate iou
    intersection = y_true * y_pred                  # |A.B|
    union        = y_true + y_pred - intersection   # |A|+|B|-|A.B|

    intersection = tf.reduce_sum(intersection, axis=0) # (c,) array
    union        = tf.reduce_sum(union, axis=0)        # (c,) array

    return intersection, union


