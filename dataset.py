import os
import shutil
from zipfile import ZipFile



def MKDIR(op='train'):
    path = '../dataset/' + op
    if not os.path.exists(path):
        os.mkdir(path)

    return


def prepareDataset(data_dir):
    ### make train/val/test directories ###
    if not os.path.exists('../dataset'):
        os.mkdir('../dataset')
    else:
        print('Already prepared dataset!')
        return

    MKDIR('train')
    MKDIR('val')
    MKDIR('test')
    train_path = os.path.join(data_dir, 'TrainData.zip')
    valid_path = os.path.join(data_dir, 'ValidData.zip')
    test_path  = os.path.join(data_dir, 'TestData.zip')
    ################# train images/labels #################
    ZipFile(train_path).extractall("../dataset/train")
    ZipFile(valid_path).extractall("../dataset/val")
    ZipFile(test_path).extractall("../dataset/test")
    # unzip train_path -d '../dataset/train'
    # unzip valod_path -d '../dataset/val'
    # unzip test_path  -d '../dataset/test'
    return


## for folder testing purpose
if __name__ == '__main__':
    print(len(os.listdir('../dataset/train/img')))
    print(len(os.listdir('../dataset/train/mask')))
    print(len(os.listdir('../dataset/val/img')))
    print(os.listdir('../dataset/test/img'))
