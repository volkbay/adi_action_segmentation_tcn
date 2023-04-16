###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# Dataloader of Kinetics400 dataset.

import os
import pickle
import torch
import numpy

from torch.utils.data import Dataset
from torchvision import transforms

class Kinetics(Dataset):
    """
    Kinetics400 Human Actions Dataset (400 action class)
    (https://deepmind.com/research/open-source/kinetics/).
    The image files are in RGB format and corresponding portrait matting files are in RGBA
    format where the alpha channel is 0 or 255 for background and portrait respectively.
    """
    def __init__(self, dir_path, split, img_size, num_classes, fold_ratio, fps, num_frames_model, num_frames_dataset, transform, data_configuration, transformVideo=False, blacklist=[]):
        self.dir_path = dir_path
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.fold_ratio = fold_ratio
        self.fps = fps
        self.num_frames_model = num_frames_model
        self.num_frames_dataset = num_frames_dataset
        self.transform = transform
        self.data_cfg = data_configuration
        self.label_distribution = numpy.zeros(num_classes)
        self.transformVideo = transformVideo
        self.blacklist = blacklist
        self.background_pickle_index = 0
        # Check split 
        if split not in ('test', 'train'):
            raise ValueError("Split name can only be set to 'test' or 'train'")   
        self.__load_dataset() 

    # Main dataloader function in the beginning
    def __load_dataset(self):
        self.dataset = []
        # Different datasets in different folders
        if self.data_cfg['fixedDataset']: # Fixed frame dataset, bulk pickles
            self.folder_name = f'processed_4class_fixed_{self.num_frames_dataset}frames_{self.img_size[0]}x{self.img_size[1]}'
        elif self.data_cfg['bulkPickles']: # Not fixed frame dataset, bulk pickles
            self.folder_name = f'processed_4class_{self.fps}fps_{self.num_frames_dataset}frames_{self.img_size[0]}x{self.img_size[1]}'            
        else: # Not fixed frame dataset, single pickles
            self.folder_name = f'processed_4class'    

        self.folder_path = os.path.join(self.dir_path, self.folder_name, self.split) 
        dir = sorted(os.listdir(self.folder_path))
        dir = [x for x in dir if x.endswith('.pkl')] # Take only pickles files
        if self.split == 'test' and (self.num_classes < 5 or self.data_cfg['singleBackgroundPickle']):
            dir = dir[:-1] # If training w/o negative label or w/ new negative samples(in /new_background), do not include test_background pickle in /test         
        print("I - ==========", self.split.upper(), " SET ==========")
        if self.data_cfg['loadData2memory']: # Load pkl to memory    
            for pickle_filename in dir[0:self.data_cfg['dataCount']]:                
                print(f'I - Loading file: {pickle_filename} in {self.folder_path}')
                pickle_filepath = os.path.join(self.folder_path, pickle_filename)
                with open(pickle_filepath, 'rb') as f:
                    dataset = pickle.load(f)
                    self.add_data_w_multiplication(dataset) if self.data_cfg['multiplyData'] else self.add_data_wo_multiplication(dataset)
            self.data_wo_background = len(self.dataset) # Single pickle mode related, stores non-background sample count
        else: # Load pkl at each getitem
            if self.data_cfg['fixedDataset']:
                print("E - Not loading bulk pickles (fixed dataset) to memory is not applicable !")
            elif self.data_cfg['bulkPickles']:
                print("E - Not loading bulk pickles to memory is not applicable !")
            elif self.data_cfg['multiplyData']:
                print("E - Cannot multiply data that is not loaded into memory !")
            else:
                for pickle_filename in dir[0:self.data_cfg['dataCount']]:
                    print(f'I - Loading file: {pickle_filename} in {self.folder_path}')
                    pickle_filepath = os.path.join(self.folder_path, pickle_filename)
                    with open(pickle_filepath, 'rb') as f:
                        (lab, imgs) = pickle.load(f)
                        if self.data_cfg["tossFirstLastFrames"]: imgs = imgs[1:-1]
                        l = len(imgs)
                        if l<self.num_frames_model: # Check sufficient frame count
                            print("I - Tossed a data with insufficient frame number.")
                        else:
                            self.dataset.append(pickle_filepath) # Append only the name of pkl file
                            self.label_distribution[lab] += 1
                            if lab in self.data_cfg["doubleClasses"] and self.split == 'train': # Doubled classes for train set
                                self.dataset.append(pickle_filepath)
                                self.label_distribution[lab] += 1
    
    # Size of dataset
    def __len__(self):
        return len(self.dataset)

    # Item loader during epochs
    def __getitem__(self, index):
        # Different type of dataset loads differently
        if self.data_cfg['loadData2memory']:
            if self.data_cfg['fixedDataset']:
                (imgs, lab, vidx) = self.dataset[index]
                index_out = vidx
            else:
                (imgs, lab) = self.dataset[index]
                index_out = index
        else:
            with open(self.dataset[index], 'rb') as f:
                (lab, imgs) = pickle.load(f)
                if self.data_cfg["tossFirstLastFrames"]: imgs = imgs[1:-1]

        start_ind = numpy.random.randint(low=0, high=(len(imgs)-self.num_frames_model+1)) # Randomly pick a frameModelNo long sequence
        images = imgs[start_ind:start_ind+self.num_frames_model]
        images = [self.fold_image(self.__normalize_image(img), self.fold_ratio) for img in images] # Normalize and fold images
        
        if self.transform is not None:
            if self.transformVideo: # Used for other models (SlowFast, X3D)
                images = numpy.array(images).transpose((3,0,1,2))  
                images_final = self.transform(torch.FloatTensor(images))    
            else: # Apply given transform
                images_transformed = [self.transform(img) for img in images]
                images_list = [img.numpy() for img in images_transformed]
                images_final = torch.Tensor(numpy.array(images_list))
        else: # No transform
            images_list = images
            images_final = torch.Tensor(numpy.array(images_list).transpose((0,3,1,2)))
        return images_final, torch.tensor(lab, dtype=torch.long), index_out

    @staticmethod
    def __normalize_image(image):
        return image / 255

    @staticmethod
    def fold_image(img, fold_ratio):
        """Folds high resolution H-W-3 image h-w-c such that H * W * 3 = h * w * c.
           These correspond to c/3 downsampled images of the original high resolution image."""
        if fold_ratio == 1:
            img_folded = img
        else:
            img_folded =numpy.empty((img.shape[0]//fold_ratio, img.shape[1]//fold_ratio, img.shape[2]*fold_ratio*fold_ratio), dtype=img.dtype)
            for i in range(fold_ratio):
                 for j in range(fold_ratio):
                    ch_idx = (i*fold_ratio + j) * img.shape[2]
                    img_folded[:, :, ch_idx:(ch_idx+img.shape[2])] = img[i::fold_ratio, j::fold_ratio, :]
        return img_folded
    
    # Append dataset with multiple samples from a single data (non-overlapping random frame sequences)
    def add_data_w_multiplication(self, dataset, blacklist_flag=True):
        if self.data_cfg['bulkPickles']:
            for data in dataset:
                # Different type of dataset loads differently
                if self.data_cfg['fixedDataset']:
                    (imgs, lab, vidx) = data
                    if vidx in self.blacklist and blacklist_flag: continue # Blacklist sample     
                else:
                    (imgs, lab) = data                    
                if len(imgs) > self.num_frames_dataset: # Check correct frame count
                    print("I - Number of frames greater than dataset description, tossed data with #frames = ", len(imgs))
                    continue
                if self.data_cfg["tossFirstLastFrames"]: imgs = imgs[1:-1]
                l = len(imgs)
                if l<self.num_frames_model: # Check sufficient frame count
                    print(f'I - Tossed a data with insufficient frame number ({l}).')
                else:
                    aug_factor = int(l/self.num_frames_model) # How many samples can be gathered from the given data
                    next_start_ind = 0
                    for n in range(aug_factor):
                        free_slot = l - (aug_factor-n)*self.num_frames_model + 1 # Limit for random frame index
                        start_ind = numpy.random.randint(low=next_start_ind, high=free_slot) # Get random starting index
                        next_start_ind = start_ind + self.num_frames_model # Next starting index randomization limit
                        if self.data_cfg['fixedDataset']:
                            self.dataset.append((imgs[start_ind:start_ind+self.num_frames_model], lab, vidx))
                        else:
                            self.dataset.append((imgs[start_ind:start_ind+self.num_frames_model], lab))
                        self.label_distribution[lab] += 1
                        if lab in self.data_cfg["doubleClasses"] and self.split == 'train': # Doubled classes for train set
                            if self.data_cfg['fixedDataset']:
                                self.dataset.append((imgs[start_ind:start_ind+self.num_frames_model], lab, vidx))
                            else:
                                self.dataset.append((imgs[start_ind:start_ind+self.num_frames_model], lab))
                            self.label_distribution[lab] += 1
        else:
            (lab, imgs) = dataset
            if len(imgs) > self.num_frames_dataset: # Check correct frame count
                print("I - Number of frames greater than dataset description, tossed data with #frames = ", len(imgs))
                return
            if self.data_cfg["tossFirstLastFrames"]: imgs = imgs[1:-1]
            l = len(imgs)
            if l<self.num_frames_model: # Check sufficient frame count
                print("I - Tossed a data with insufficient frame number.")
            else:
                aug_factor = int(l/self.num_frames_model) # How many samples can be gathered from the given data
                next_start_ind = 0
                for n in range(aug_factor):
                    free_slot = l - (aug_factor-n)*self.num_frames_model + 1 # Limit for random frame index
                    start_ind = numpy.random.randint(low=next_start_ind, high=free_slot) # Get random starting index
                    next_start_ind = start_ind + self.num_frames_model # Next starting index randomization limit
                    self.dataset.append((imgs[start_ind:start_ind+self.num_frames_model], lab))
                    self.label_distribution[lab] += 1
                    if lab in self.data_cfg["doubleClasses"] and self.split == 'train': # Doubled classes for train set
                        self.dataset.append((imgs[start_ind:start_ind+self.num_frames_model], lab))
                        self.label_distribution[lab] += 1

    # Append dataset with a singe sample from a single data (random starting point frame sequence)         
    def add_data_wo_multiplication(self, dataset, blacklist_flag=True):
        if self.data_cfg['bulkPickles']:
            for data in dataset:
                # Different type of dataset loads differently
                if self.data_cfg['fixedDataset']:
                    (imgs, lab, vidx) = data
                    if vidx in self.blacklist and blacklist_flag: continue # Blacklist sample
                else:
                    (imgs, lab) = data
                if len(imgs) > self.num_frames_dataset: # Check correct frame count
                    print("I - Number of frames greater than dataset description, tossed data with #frames = ", len(imgs))
                    continue
                if self.data_cfg["tossFirstLastFrames"]: imgs = imgs[1:-1]
                l = len(imgs)
                if l<self.num_frames_model: # Check sufficient frame count
                    print("I - Tossed a data with insufficient frame number.")
                else:
                    if self.data_cfg['fixedDataset']:
                        self.dataset.append((imgs, lab, vidx))
                    else:
                        self.dataset.append((imgs, lab))
                    self.label_distribution[lab] += 1
                    if lab in self.data_cfg["doubleClasses"] and self.split == 'train': # Doubled classes for train set
                        if self.data_cfg['fixedDataset']:
                            self.dataset.append((imgs, lab, vidx))
                        else:
                            self.dataset.append((imgs, lab))
                        self.label_distribution[lab] += 1
        else:
            (lab, imgs) = dataset
            if len(imgs) > self.num_frames_dataset: # Check correct frame count
                print("I - Number of frames greater than dataset description, tossed data with #frames = ", len(imgs))
                return            
            if self.data_cfg["tossFirstLastFrames"]: imgs = imgs[1:-1]
            l = len(imgs)
            if l<self.num_frames_model: # Check sufficient frame count
                print("I - Tossed a data with insufficient frame number.")
            else:
                self.dataset.append((imgs, lab))
                self.label_distribution[lab] += 1
                if lab in self.data_cfg["doubleClasses"] and self.split == 'train': # Doubled classes for train set
                    self.dataset.append((imgs, lab))
                    self.label_distribution[lab] += 1
    
    # Single pickle mode, background data loader (this is called at the beginning of each epoch)
    def load_next_background_pickle(self):    
        if len(self)>self.data_wo_background: # Delete current background data  
            del self.dataset[self.data_wo_background:]
            self.label_distribution[-1] = 0 
        if self.data_cfg['fixedDataset'] and self.data_cfg['bulkPickles'] and self.data_cfg['loadData2memory']:
            folder_path = os.path.join(self.dir_path, self.folder_name, self.split, self.data_cfg['singleBackgroundPath'])
            dir = sorted(os.listdir(folder_path))
            dir = [x for x in dir if x.endswith('.pkl')] # Take only pkl files
            ind = self.background_pickle_index % len(dir) # Circular shift to next background pkl index
            self.background_pickle_index += 1
            pickle_filename = dir[ind]
            print(f'I - Loading file: {pickle_filename} in {folder_path}')
            pickle_filepath = os.path.join(folder_path, pickle_filename)
            with open(pickle_filepath, 'rb') as f:
                dataset = pickle.load(f)
                # Add data without blacklisting
                self.add_data_w_multiplication(dataset, blacklist_flag=False) if self.data_cfg['multiplyData'] else self.add_data_wo_multiplication(dataset, blacklist_flag=False)
            print("I - New label distribution:", self.label_distribution)
            print('')
        else:
            print('E - Configuration is not compatible with single background pickle mode.')
