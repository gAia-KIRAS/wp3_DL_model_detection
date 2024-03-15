class HyperParameter(object):
  def __init__(self):
    self.label_dict    = dict( background = 0,
                               landslide  = 1)

    # Parameter for generator and training
    self.H             = 128    # Height
    self.W             = 128    # Width
    self.C             = 19     # Channel
    self.batch_size    = 20 # 12
    self.start_batch   = 0
    self.learning_rate = 4e-4
    self.is_aug        = True
    self.class_num     = 2
    self.epoch_num     = 273

    ####  self.img_mean is provided in forum, the real max (self.img_max) is deviced by self.img_mean 
    self.img_mean      = [1111.81236406, 824.63171476, 663.41636217, 445.17289745,
                          645.8582926, 1547.73508126, 1960.44401001, 1941.32229668,
                          674.07572865, 9.04787384, 1113.98338755, 519.90397929,
                          20.29228266, 772.83144788]   # means of 14 bands
    self.img_max       = [3453., 16296., 20932., 14762.,
                          6441.,  6414.,  7239., 16138.,  2392.,
                          194., 6446., 10222.,
                          82.,  3958.]    # real max data before dividing by imag_mean

    #### extract from h5 file --> these number are divided by self.img_mean before releasing officially
    self.mean_af = [0.92570402, 0.92270051, 0.95410915, 0.95963868, 1.02278967, 1.04261156,
                   1.0358439,  1.04675552, 1.16994136, 1.17359787, 1.04949727, 1.03703151,
                   1.25110812, 1.64954336]
    self.min_af =  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    self.max_af = [ 3.10573988, 19.76154895, 31.55182958, 33.16014988,  9.97277588,  4.14412006,
                    3.69253086,  8.31289067,  3.54856272, 21.44150144,  5.78644177, 19.66132287,
                    4.04094509,  5.1214272 ]


    self.gaia_mean = [241.9, 285.7, 239.1, 926.9, 585.7]
