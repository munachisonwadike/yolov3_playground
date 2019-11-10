from __future__ import print_function, division


import numpy as np
import os
import random
import torch


from dataloaders import custom_transforms as tr
from mypath import Path
from PIL import Image
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset


sample_train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']


class VaihingenSegmentation(Dataset):
    """
    Vahingen dataset
    """

    test_ids = ['5', '21', '15', '30']


    # Parameters
    WINDOW_SIZE = (256, 256) # Patch size
    STRIDE = 32 # Stride for testing
    IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
    FOLDER = "/home/mmvc/mmvc-ny-local/Munachiso_Nwadike/ISPRS_DATASET/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
    BATCH_SIZE = 4 # Number of samples in a mini-batch

    LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
    N_CLASSES = len(LABELS) # Number of classes
    NUM_CLASSES  = N_CLASSES
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory

    DATASET = 'Vaihingen'

    if DATASET == 'Potsdam':
        MAIN_FOLDER = FOLDER + 'Potsdam/'
        DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_{}_IRRG.tif'
        LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_{}_label.tif'
        ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'    
    elif DATASET == 'Vaihingen':
        MAIN_FOLDER = FOLDER + 'Vaihingen/'
        DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
        LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
        ERODED_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}_noBoundary.tif'


    # ISPRS color palette
    # Let's define the standard ISPRS color palette
    palette = {0 : (255, 255, 255), # Impervious surfaces (white)
               1 : (0, 0, 255),     # Buildings (blue)
               2 : (0, 255, 255),   # Low vegetation (cyan)
               3 : (0, 255, 0),     # Trees (green)
               4 : (255, 255, 0),   # Cars (yellow)
               5 : (255, 0, 0),     # Clutter (red)
               6 : (0, 0, 0)}       # Undefined (black)

    invert_palette = {v: k for k, v in palette.items()}


    #START REPLACEMENT
  
    def __init__(self, ids, data_files=None, label_files=None,
                            cache=False, augmentation=True):
        super(VaihingenSegmentation, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        
        # List of files
        self.data_files = [self.DATA_FOLDER.format(id) for id in ids]
        self.label_files = [self.LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000


        # Utils

    def get_random_pos(self, img, window_shape):
        """ Extract of 2D random patch of shape window_shape in the image """
        w, h = window_shape
        W, H = img.shape[-2:]
        x1 = random.randint(0, W - w - 1)
        x2 = x1 + w
        y1 = random.randint(0, H - h - 1)
        y2 = y1 + h
        return x1, x2, y1, y2

    def CrossEntropy2d(self, input, target, weight=None, size_average=True):
        """ 2D version of the cross entropy loss """
        dim = input.dim()
        if dim == 2:
            return F.cross_entropy(input, target, weight, size_average)
        elif dim == 4:
            output = input.view(input.size(0),input.size(1), -1)
            output = torch.transpose(output,1,2).contiguous()
            output = output.view(-1,output.size(2))
            target = target.view(-1)
            return F.cross_entropy(output, target,weight, size_average)
        else:
            raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

    def accuracy(self, input, target):
        return 100 * float(np.count_nonzero(input == target)) / target.size

    def sliding_window(self, top, step=10, window_size=(20,20)):
        """ Slide a window_shape window across the image with a stride of step """
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                yield x, y, window_size[0], window_size[1]
                
    def count_sliding_window(self, top, step=10, window_size=(20,20)):
        """ Count the number of windows in an image """
        c = 0
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                c += 1
        return c

    def grouper(self, n, iterable):
        """ Browse an iterator by chunk of n elements """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk

    def metrics(self, predictions, gts, label_values=LABELS):
        cm = confusion_matrix(
                gts,
                predictions,
                range(len(label_values)))
        
        print("Confusion matrix :")
        print(cm)
        
        print("---")
        
        # Compute global accuracy
        total = sum(sum(cm))
        accuracy = sum([cm[x][x] for x in range(len(cm))])
        accuracy *= 100 / float(total)
        print("{} pixels processed".format(total))
        print("Total accuracy : {}%".format(accuracy))
        
        print("---")
        
        # Compute F1 score
        F1Score = np.zeros(len(label_values))
        for i in range(len(label_values)):
            try:
                F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
            except:
                # Ignore exception if there is no element in class i for test set
                pass
        print("F1Score :")
        for l_id, score in enumerate(F1Score):
            print("{}: {}".format(label_values[l_id], score))

        print("---")
            
        # Compute kappa coefficient
        total = np.sum(cm)
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
        kappa = (pa - pe) / (1 - pe);
        print("Kappa: " + str(kappa))
        return accuracy


    def convert_to_color(self, arr_2d, palette=palette):
        """ Numeric labels to RGB-color encoding """
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in palette.items():
            m = arr_2d == c
            arr_3d[m] = i

        return arr_3d

    def convert_from_color(self, arr_3d, palette=invert_palette):
        """ RGB-color encoding to grayscale labels """
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d

    # @classmethod
    def data_augmentation(self, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1/255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # Labels are converted from RGB to their numeric values
            label = np.asarray(self.convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = self.get_random_pos(data, self.WINDOW_SIZE)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        _img, _target = data_p, label_p

        sample = {'image': _img, 'label': _target}
        # Return the torch.Tensor values
        return sample


    # END REPLACEMENT 


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.base_size = 513
    # args.crop_size = 513

    vai_train = VaihingenSegmentation(sample_train_ids, cache=CACHE)

    dataloader = DataLoader(vai_train, batch_size=1, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            # img_tmp *= (0.229, 0.224, 0.225)
            # img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


