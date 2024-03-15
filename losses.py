import tensorflow as tf
import math
from datetime import datetime
from scipy.ndimage import distance_transform_edt as distance

#####################################################################################
#####   All of losses were coded to be optimized for gpu parallel computation   #####
#####################################################################################


class EntropyLoss(tf.keras.losses.Loss):
    def __init__(self, scale=1., num_classes=2, name='EntropyLoss', **kwargs):
        super(EntropyLoss, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
        # Compute loss on batch
            # Input:
                - y_true (3d tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4d tf tensor): BxWxHxC - predicted class probabilities (already apply softmax)
            # Output:
                - entropy loss: a scalar
        """
        ## one-hot
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        ## avoid absolute zero and absolute one
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)

        ## cross entropy loss
        ce_loss = tf.cast((- y_true * tf.math.log(y_pred)), dtype=tf.float32) # BxWxHxC

        ## downgrade background class(represented by 0) by scale times (0 <= scale <= 1)
        alpha   = tf.ones_like(y_pred[:,:,:,:1]) * self.scale # BxWxHx1
        alpha   = tf.concat([alpha, tf.ones_like(y_pred[:,:,:,1:])], axis=-1)   # BxWxHxC

        ## total loss
        entropy_loss   = alpha * ce_loss
        entropy_loss   = tf.reduce_mean(entropy_loss)

        return tf.cast(entropy_loss, dtype=tf.float32)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., scale=1., num_classes=2, name='FocalLoss', **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.gamma = gamma
        self.scale = scale
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
            # Compute loss on batch
            # Input:
                - y_true (3d tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4d tf tensor): BxWxHx2 - predicted class probabilities (already apply softmax)
            # Output:
                - focal loss: a scalar
        """
        ## one-hot
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        ## avoid absolute zero and absolute one
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)

        ## cross entropy loss
        ce_loss = tf.cast((- y_true * tf.math.log(y_pred)), dtype=tf.float32) # BxWxHxC

        ## downgrade background class(represented by 0) by scale times (0 <= scale <= 1)
        alpha = tf.ones_like(y_pred[:,:,:,:1]) * self.scale # BxWxHx1
        alpha = tf.concat([alpha, tf.ones_like(y_pred[:,:,:,1:])], axis=-1)   # BxWxHxC

        ## focal weight
        focal_weight = tf.where(y_true == 1, 1 - y_pred, y_pred)
        focal_weight = tf.cast(focal_weight, dtype=tf.float32)
        focal_weight = alpha * (focal_weight ** self.gamma)

        ## focal loss
        focal_loss   = focal_weight * ce_loss
        focal_loss   = tf.reduce_mean(focal_loss)

        return tf.cast(focal_loss, dtype=tf.float32)


class LogCoshLoss(tf.keras.losses.Loss):
    def __init__(self, scale=.8, num_classes=2, name='LogCoshLoss', **kwargs):
        super(LogCoshLoss, self).__init__(name=name, **kwargs)
        self.scale = scale
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
            # Compute loss on batch
            # Input:
                - y_true (3d tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4d tf tensor): BxWxHxC - predicted class probabilities (already apply softmax)
            # Output:
                - log cosh loss: a scalar
        """
        ## one-hot
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        ## avoid absolute zero and absolute one
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)

        ## downgrade background class(represented by 0) by scale times (0 <= scale <= 1)
        alpha   = tf.ones_like(y_pred[:,:,:,:1]) * self.scale # BxWxHx1
        alpha   = tf.concat([alpha, tf.ones_like(y_pred[:,:,:,1:])], axis=-1)   # BxWxHxC

        ## log cosh loss
        dif = tf.cast((y_pred - y_true)**2, dtype=tf.float32) # BxWxHxC
        logcosh_loss = tf.math.log(tf.math.cosh(dif))
        logcosh_loss = alpha * logcosh_loss
        logcosh_loss = tf.reduce_mean(logcosh_loss)

        return tf.cast(logcosh_loss, dtype=tf.float32)


class IOULoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=2, name='IOULoss', **kwargs):
        super(IOULoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
            # Calculate IOU loss on batch for multi-class segmentation.
            # Input:
                - y_true (3D tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4D tf tensor): BxWxHxC - predicted class probabilities (already apply softmax)
            # Output:
                - iou loss: a scalar
        """
        ## prevent division by zero
        epsilon = 1e-7

        ## one-hot
        y_true  = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        ## reshape
        y_true  = tf.reshape(y_true, [-1, self.num_classes]) # (B*W*H,C)
        y_pred  = tf.reshape(y_pred, [-1, self.num_classes]) # (B*W*H,C)

        ## calculate iou loss of each class
        intersection = tf.reduce_sum(y_true*y_pred, axis=0)                # |A| n |B|
        union        = tf.reduce_sum(y_true+y_pred, axis=0) - intersection # |A| u |B| - (|A|n|B|)

        iou = (intersection + epsilon) / (union + epsilon)

        ## calculate the IOU loss
        iou_loss = -tf.math.log(iou) #  can use -log(iou) or 1 - iou
        iou_loss = tf.reduce_mean(iou_loss)

        return tf.cast(iou_loss, dtype=tf.float32)


class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.75, beta=0.75, num_classes=2, name='TverskyLoss', **kwargs):
        super(TverskyLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.beta  = beta
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
            # calculate on batch
            # input:
                - y_true (3d tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4d tf tensor): BxWxHxC - class probabilities at each prediction (after softmax)
            # output:
                - tversky loss: a scalar
        """
        ## prevent division by zero
        epsilon = 1e-7

        ## one-hot
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        ## reshape
        y_true = tf.reshape(y_true, [-1, self.num_classes]) # (B*W*H,C)
        y_pred = tf.reshape(y_pred, [-1, self.num_classes]) # (B*W*H,C)

        ## calculate true positives, false positives, and false negatives
        tp = tf.reduce_sum(y_true * y_pred, axis=0)    # |A| n |B|
        fp = tf.reduce_sum((1-y_true)*y_pred, axis=0)  # |A| \ |B|
        fn = tf.reduce_sum(y_true*(1-y_pred), axis=0)  # |B| \ |A|

        ## calculate the Tversky coefficient
        tversky_coeff = (tp + epsilon) / (tp + self.alpha*fp + self.beta*fn + epsilon)

        ## calculate the Tversky loss
        tversky_loss = -tf.math.log(tversky_coeff) # can use -log(tversky_coeff) or 1 - tversky_coeff
        tversky_loss = tf.reduce_mean(tversky_loss)

        return tf.cast(tversky_loss, dtype=tf.float32)


class LovaszLoss(tf.keras.losses.Loss):
    def __init__(self, is_per_img=False, num_classes=2, name='LovaszLoss', **kwargs):
        super(LovaszLoss, self).__init__(name=name, **kwargs)
        self.is_per_img = is_per_img
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
            # Calculate on batch
            # Input:
                - y_true (3d tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4d tf tensor): BxWxHxC - predicted class probabilities (already apply softmax)
            # Output:
                - lovasz loss: a scalar
        """
        ## one-hot
        y_true  = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        B,W,H,C = y_true.shape
        ## reshape
        y_true  = tf.reshape(y_true, (-1,self.num_classes)) # (B*W*H)*C
        y_pred  = tf.reshape(y_pred, (-1,self.num_classes)) # (B*W*H)*C
        
        ## Lovazs loss per image or per batch
        if self.is_per_img:
            losses = []
            n_imgs = y_true.shape[0] / (W*H)
            for i in range(n_imgs): # per image
                end_i = (i+1)*(W*H) 
                loss = self.calulateLovaszLoss(y_pred[i:end_i], y_true[i:end_i])
                losses.append(loss)
            loss = tf.reduce_mean(losses)
        else:
            loss = self.calulateLovaszLoss(y_pred, y_true)

        return tf.cast(loss, dtype=tf.float32)

    def calulateLovaszLoss(self, probas, labels):
        """
            # Calculate per batch
            # Input:
                - probas (2d tensor): (B*W*H)xC - class probabilities at each prediction (after softmax)
                - labels (2d tensor): (B*W*H)xC - ground truth labels (integer values) - 0 or 1
            # Output:
                - loss of several images
        """
        losses = []
        for c in range(self.num_classes):
            class_preds  = probas[..., c]
            class_labels = labels[..., c]

            ## avoid absolute zero and absolute one
            class_preds  = tf.clip_by_value(class_preds, 1e-7, 1. - 1e-7)

            errors       = tf.math.abs(class_labels - class_preds)
            errors       = tf.math.sqrt(errors)

            errors_sorted, perm = tf.nn.top_k(errors, k=tf.size(errors), name="descending_sort")
            gt_sorted = tf.gather(class_labels, perm)

            grad = self.calculateLovaszGrad(gt_sorted)
            grad = tf.stop_gradient(grad)
            errors_sorted = tf.nn.relu(errors_sorted)

            loss = tf.tensordot(errors_sorted, grad, axes=1, name="loss_non_void") # matrix multiplication (dot product)
            losses.append(loss) # a scalar

        loss = tf.reduce_mean(losses) # across classes

        return loss

    def calculateLovaszGrad(self, gt_sorted):
        """
            # Calculate per class of an image
            # Input:
                - gr_sorted (2d tensor): WxH - ground truth labels (integer values) already in sorted prediction probability order
            # Output:
                -
        """
        ## avoid divided by 0
        epsilon = 1e-7

        gts = tf.reduce_sum(gt_sorted)

        intersection = gts - tf.cumsum(gt_sorted)
        union        = gts + tf.cumsum(1. - gt_sorted)

        iou = (intersection + epsilon) / (union + epsilon)
        jaccard = -tf.math.log(iou) # 1. - intersection / union
        jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)

        return jaccard


class BoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, scale=0.8, num_classes=2, name='BoundaryLoss', **kwargs):
        super(BoundaryLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.scale = scale

    def call(self, y_true, y_pred):
        """
            # Compute loss on batch
            # Input:
                - y_true (3d tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4d tf tensor): BxWxHxC - predicted class probabilities (already apply softmax)
            # Output:
                - boundary loss: a scalar
        """
        ## boundary weight
        boundary_weight = self.calculateDistMap(y_true)

        ## avoid absolute zero and absolute one
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7) # BxWxHxC

        ## one-hot
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        ## Compute the mean loss across classes
        boundary_loss = boundary_weight * (- y_true * tf.math.log(y_pred))
        boundary_loss = tf.reduce_mean(boundary_loss)

        return tf.cast(boundary_loss, dtype=tf.float32)

    def calculateDistMap(self, y_true):
        dist_map = []
        for i in range(y_true.shape[0]): # each image in a batch of image
            posmask = y_true[i]
            negmask = 1 - posmask
            if tf.reduce_sum(posmask) != 0 and tf.reduce_sum(negmask) != 0: # case of not only background (some foreground pixel)
                posmask_dist = tf.convert_to_tensor(distance(posmask)+1, dtype=tf.float32)
                negmask_dist = tf.convert_to_tensor(distance(negmask)+1, dtype=tf.float32)

                posmax  = tf.reduce_max(posmask_dist)
                negmax  = tf.reduce_max(negmask_dist)

                posmask = posmask*posmask_dist
                negmask = negmask*negmask_dist

                posmask = tf.where(tf.math.logical_and(posmask > 0, posmask < 4.5), posmax, 1.)
                negmask = tf.where(tf.math.logical_and(negmask > 0, posmask < 4.5), negmax, 1.) * self.scale

                posmask = tf.expand_dims(posmask, axis=-1)
                negmask = tf.expand_dims(negmask, axis=-1)

                res  = tf.concat([negmask, posmask], axis=-1)
            else: # only one class -> distance will be all 1
                ones = tf.ones_like(posmask)
                ones = tf.expand_dims(ones, axis=-1)
                res  = tf.concat([ones, ones], axis=-1)

            dist_map.append(res)

        dist_map = tf.convert_to_tensor(dist_map, dtype=tf.float32) # BxWxH

        return dist_map


class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=2, name='CenterLoss', **kwargs):
        super(CenterLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        """
            # Compute loss on batch
            # Input:
                - y_true (3d tf tensor): BxWxH - ground truth labels (integer values)
                - y_pred (4d tf tensor): BxWxHxC - predicted class probabilities (already apply softmax)
            # Output:
                - boundary loss: a scalar
        """
        ## boundary weight
        boundary_weight = self.calculateDistMap(y_true)

        ## avoid absolute zero and absolute one
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7) # BxWxHxC

        ## one-hot
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.uint8), depth=self.num_classes, axis=-1) # BxWxHxC

        ## Compute the mean loss across classes
        boundary_loss = boundary_weight * (- y_true * tf.math.log(y_pred))
        boundary_loss = tf.reduce_mean(boundary_loss)

        return tf.cast(boundary_loss, dtype=tf.float32)

    def calculateDistMap(self, y_true):
        dist_map = []
        for i in range(y_true.shape[0]): # each image in a batch of image
            posmask = y_true[i]
            negmask = 1 - posmask
            if tf.reduce_sum(posmask) != 0 and tf.reduce_sum(negmask) != 0: # case of not only background (some foreground pixel)
                posmask_dist = tf.convert_to_tensor(distance(posmask)+1, dtype=tf.float32)
                negmask_dist = tf.convert_to_tensor(distance(negmask)+1, dtype=tf.float32)

                posmask = tf.expand_dims(posmask, axis=-1)
                negmask = tf.expand_dims(negmask, axis=-1)

                res  = tf.concat([negmask, posmask], axis=-1)
            else:
                ones = tf.ones_like(posmask)
                ones = tf.expand_dims(ones, axis=-1)
                res  = tf.concat([ones, ones], axis=-1)

            dist_map.append(res)

        dist_map = tf.convert_to_tensor(dist_map, dtype=tf.float32) # BxWxH

        return dist_map
