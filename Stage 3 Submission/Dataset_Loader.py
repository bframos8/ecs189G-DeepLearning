'''
Concrete IO class for a specific dataset
'''

# Based off of Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>

import pickle
import numpy as np
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        raw = pickle.load(f)
        f.close()

        def unpack(split):
            # for ORL
            # shift = 1
            # for MNIST and CIFAR
            shift = 0
            images, labels = [], []
            for inst in raw[split]:
                img = inst['image']  # shape either (H,W) or (H,W,3)
                lab = inst['label'] - shift

                if img.ndim == 3 and img.shape[2] == 3:
                    # CIFAR‐10 color image: move channel to axis=0
                    img = np.transpose(img, (2, 0, 1))  # now (3,H,W)
                else:
                    # grayscale: drop any trailing channel
                    img = img if img.ndim == 2 else img[:, :, 0]
                    # make it explicit channel‐first
                    img = img[np.newaxis, :, :]  # (1,H,W)

                images.append(img.astype(np.float32) / 255.0)
                labels.append(lab)
            return {'image': images, 'label': labels}

        self.data = {
            'train': unpack('train'),
            'test': unpack('test')
        }
        return self.data