###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# Creates .pkl binary files from video samples. Spans dir_names and creates pickles correspondingly.
# If 1000 samples are reaced per pickle, start writing next .pkl. This code samples fixed number of 
# frame (dataset_frame_no) from the complete video.

import os
import pickle
import cv2
import numpy as np

# Frame Parameters
img_size = [256, 256]
dataset_frame_no = 50

# Paths and making folders
class_names = ['pull ups', 'push up', 'situp', 'squat', 'background', 'test', 'test_background'] # Class list. DO NOT change.
dir_names = ['background'] # ENTER desired classes to be pickle from the list class_names
path = '/data_ssd/processed/kinetics400/' # Linux path
# path = 'C:\\prj\\tcn\\dat\\kinetics400' # Windows path
pickles_folder = f'pickles'
pickles_folder_path = os.path.join(path, pickles_folder)
os.makedirs(pickles_folder_path, exist_ok=True)

# Center crop the frames regarding the short edge
def adjust_img(image, target_img_size):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize according to short edge
    img_size = image.shape
    rat0 = target_img_size[0] / img_size[0]
    rat1 = target_img_size[1] / img_size[1]
    resize_ratio = max(rat0, rat1)
    img_resized = cv2.resize(img, (0,0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)
    # Crop the middle of resized image
    min_x = (img_resized.shape[0] - target_img_size[0]) // 2
    max_x = min_x + target_img_size[0]
    min_y = (img_resized.shape[1] - target_img_size[1]) // 2
    max_y = min_y + target_img_size[1]
    img_resized = img_resized[min_x:max_x, min_y:max_y, :]
    if img_resized.shape == (target_img_size[0], target_img_size[1], 3): # Check correct size
        return True, img_resized
    else:
        return False, None

# Write pkl file with index and class name
def write_pickle(class_name, class_no, dataset, pickle_idx, path):
    class_name = class_name.replace(' ','_')
    num_vids = len(dataset)
    if class_no < 5: # Train pickles
        file = f'dataset_cls{class_no}_{class_name}{pickle_idx:02d}_no_samples{num_vids}.pkl'
    else: # Test pickles
        file = f'dataset_{class_name}{pickle_idx:02d}_no_samples{num_vids}.pkl'
    with open(os.path.join(path, file), "wb") as output_file:
        print(f'I - Writing pickle {file}')
        pickle.dump(dataset, output_file)  

# Main block
for cls in dir_names:
    cls_no = class_names.index(cls)
    cls_path = os.path.join(path, cls)
    dataset = []
    pickle_idx = 0
    print(f'********************---------- CLASS: "{cls.upper()}" ----------********************')
    for vid in sorted(os.listdir(cls_path)):
        if vid.endswith('.mp4'): # Check correct file type
            retry = False # Retrial flag if cv2.frame_count is not equal to the actual number of frames
            first_pass = True # First trial flag of the current video sample
            while(retry or first_pass):
                first_pass = False
                vid_path = os.path.join(cls_path, vid)
                cap = cv2.VideoCapture(vid_path)                
                if cap.isOpened():
                    vidW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    vidH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    
                    if retry:
                        vidF = frame_counter # Actual number of frames
                        retry = False
                    else:
                        vidF = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Number of frames given by cv2
                    
                    vidFrames = []
                    if vidF < dataset_frame_no:
                        print(f'W - Insufficient frame number ({vidF}<{dataset_frame_no}) skipping {vid}')
                    elif vidF > 1000:
                        print(f'W - Too many frames ({vidF}) skipping {vid}')
                    else:                    
                        print(f'I - Reading {vid} with {vidF} frames')
                        frame_idx = np.linspace(1, vidF, dataset_frame_no, dtype=np.uint32) # Sample fixed number of frame from whole video
                        frame_counter = 1

                        # Start sampling frames
                        is_read, frame = cap.read()
                        while is_read:
                            if frame_counter in frame_idx:                        
                                is_resized, frame_resized = adjust_img(frame, img_size)                      
                                if is_resized: vidFrames.append(frame_resized) # Successfully resized
                                else: print(f'W - Frame {frame_counter} is not resized correctly with shape {frame_resized.shape}!')
                            is_read, frame = cap.read()
                            frame_counter += 1

                        frame_counter -= 1 # Correct number of frames, will be used if retry
                        if frame_counter < vidF:
                            print(f'W - Cannot read all the frames of {vid}, retrying.')
                            retry = True
                        elif len(vidFrames) == dataset_frame_no: # Successfully sampled
                            dataset_index = int(vid.split('_')[1]) # Video file index
                            label_str = ' '.join(vid.split('_')[2:])[:-4] # Video label
                            if label_str in class_names: # Label in training classes
                                label = class_names.index(label_str)
                            else:
                                label = class_names.index('background')
                            dataset.append((vidFrames, label, dataset_index))
                            if len(dataset) == 1000: # Write pkl if 1000 samples
                                write_pickle(cls, cls_no, dataset, pickle_idx, pickles_folder_path)
                                dataset.clear()
                                pickle_idx += 1
                        else:
                            print(f'W - Video {vid} is not sampled correctly with {len(vidFrames)} frames ')
                else:
                    print(f'W - Cannot open and skipping {vid}')
                    retry = False
                cap.release()
    if len(dataset)>0: write_pickle(cls, cls_no, dataset, pickle_idx, pickles_folder_path) # Write last pkl for current dir