###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# Get results of our Kinetics dataset of SlowFast model trained with Kinetics400

import torch
import sys
import numpy

sys.path.append('../../tcn')
import kinetics400
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

import json
import urllib

# Fixed transform values by original code
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

# Given Slowfast model is 32 frames, so cannot multiply data. Select other configuration depending on desired dataset
data_cfg = {'multiplyData': False, 'bulkPickles': True, 'dataCount': 4, 'loadData2memory': True, 'doubleClasses': [-1], 'fixedDataset': True,
            'tossFirstLastFrames': True, 'singleBackgroundPath': 'new_background', 'singleBackgroundPickle': True}
dir_path = '/data_ssd/processed/kinetics400/'
slowfast_alpha = 4 # Fixed
num_frame_model = 32 # Fixed

def compute_acc(outputs, labels):
    pred = numpy.array(outputs)
    labels_np = numpy.array(labels)
    total = len(labels)
    correct = sum(pred == labels_np)
    return total, correct

# Load model and class names
json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

# Original transfor for slowfast algorithm. Creates slow&fast frame sequences.    
class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('I - Running on device: {}'.format(device))
print("I -", end=" ")
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
model = model.eval()
model = model.to(device)

# Create dataset. ENTER desired split 'test' or 'train'
transform = transforms.Compose([NormalizeVideo(mean, std), PackPathway()])
test_set = kinetics400.Kinetics(dir_path=dir_path, split='test', img_size=(256, 256), num_classes=5, 
            fold_ratio=1, fps=5, num_frames_model=num_frame_model, num_frames_dataset=50, transform=transform,
            data_configuration = data_cfg, transformVideo=True)
if data_cfg['singleBackgroundPath']: test_set.load_next_background_pickle()
print("I - Test set length: ", str(len(test_set)))
print("I - Label distribution:", test_set.label_distribution)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# Check Dataloader Output
for data, label, ind in test_loader:
    print("I - Batch size: ", len(data[1]), " tensor shape: ", data[1][0].shape," data min-max: ", data[1].min(), data[1].max())
    print("I - Label min-max: ", label.min(), label.max(), "data number in dataset: ", ind)
    break

# Evaluation
with torch.no_grad():
    val_corr, val_tot = 0,0
    confusion_matrix_valid = torch.zeros((5, 5))
    outputs = []
    labels = []
    vidx = []
    for num_batch, (batch, label, ind) in enumerate(test_loader, 1):
        # Batch
        labels.append(label.item())
        vidx.append(ind.item())
        inputs = [i.to(device) for i in batch]
        preds = model(inputs)

        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=1).indices[0]

        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
        if pred_class_names[0] == 'pull ups':
            outputs.append(0)
        elif pred_class_names[0] == 'push up':
            outputs.append(1)
        elif pred_class_names[0] == 'situp':
            outputs.append(2)
        elif pred_class_names[0] == 'squat':
            outputs.append(3)
        else:
            outputs.append(4)
    
    # Metrics
    val_tot, val_corr = compute_acc(outputs,labels)           
    for pred, lab in zip(outputs, labels):
        confusion_matrix_valid[pred, lab] += 1
  
    print("I - num batch:", num_batch)
    val_acc = 100.*(val_corr)/val_tot
    print('I - Val -- Acc: %.3f%%' % (val_acc))
    print('I - Confusion Matrix: [row->prediction - col->label]')
    print(confusion_matrix_valid.numpy())
    print('')

    # Print results
    for n, vid in enumerate(vidx):
        print(vid, labels[n], outputs[n])