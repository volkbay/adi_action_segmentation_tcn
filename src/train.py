###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# Training script for Kinetics dataset with TCN model.

from asyncore import write
from tabnanny import check
import tcn
import ai8x
import torch
import config
import time
import numpy as np

from kinetics400 import Kinetics
from torchvision import transforms
from torchinfo import summary
from torch.utils.data import DataLoader
from distiller import apputils
from torch.utils.tensorboard import SummaryWriter

# Computes standard accuracy
def compute_acc(outputs, labels):
    pred = outputs.max(1).indices 
    total = len(labels)
    correct = pred.eq(labels).sum().item()
    return total, correct

# Computes weighted accuracy with class weights
def compute_weighted_acc(outputs, labels, cls_w):
    pred = outputs.max(1).indices
    cls_w_np = np.array(cls_w)
    weight_list = cls_w_np[labels.cpu()]
    if isinstance(weight_list, np.ndarray):
        total = sum(weight_list)
    else: # if only a single element, not an array
        total = weight_list
    correct = pred.eq(labels).cpu().numpy()*weight_list
    correct = correct.sum()
    return total, correct

# Tensorboard related, computes two kinds of histogram distributions of each class: ground truth and predicted samples
def prediction_dist(outputs, labels, num_batch, ground_list, pred_list, batch_size):
    pred = outputs.max(1).indices
    outputs_np = outputs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()    
    for n, lab in enumerate(labels_np):
        ground_list[lab, num_batch*batch_size+n] = outputs_np[n,lab]
        pred_list[pred[n], num_batch*batch_size+n] = outputs_np[n,pred[n]]
    return

# Tensorboard related, writes two kind of histograms to tensorboard every epoch 
def write_histogram(split: str, writer, epoch, ground_list, pred_list):
    num_classes = ground_list.shape[0]
    for n in range(num_classes):
        a = ground_list[n,:]
        a_values = a[a != -1] # filter default values (-1)
        b = pred_list[n,:]
        b_values = b[b != -1] # filter default values (-1) 
        if a_values.size: # If there is any non-default values in ground list
            writer.add_histogram(split+" ground truth dist/ Class "+str(n), a_values, epoch)
        if b_values.size: # If there is any non-default values in pred list
            writer.add_histogram(split+" pred dist/ Class "+str(n), b_values, epoch)
    ground_list.fill(-1) # Clear values in dist arrays
    pred_list.fill(-1)
    return

# Print prediction results [video id-ground truth-prection-soft result] neatly (10 samples in a line)
def print_neat(x):
    for n in range(x.shape[0]):
        print("[",end='')
        for m in range(x.shape[1]):
            if m == x.shape[1]-1:
                print(f'{x[n,m]:.2f}', end=']')
            else:
                print(f'{int(x[n,m])}', end='-')
        if (n+1)%10 == 0: print('\n',end='')

# Main Process
if __name__ == '__main__':
    print(time.ctime())
    # Read Configuration Parameter
    cfg = config.ConfigParser()
    blk = config.ConfigParser(cfg.read('blacklistFile'))
    cfg.printAll()
    blk.printAll()
    label_names = cfg.read('labels')
    num_classes = len(label_names)
    img_size = cfg.read('imgSize')
    num_frames_model = cfg.read('frameNoModel')
    num_frames_dataset = cfg.read('frameNoDataset')
    num_epochs = cfg.read('epochNo')
    bias = cfg.read('bias')
    batch_size = cfg.read('batchSize')
    cls_weights = cfg.read('classWeights')
    cls_weights = cls_weights[0:num_classes]
    cls_weights_flag = cfg.read('classWeightsFlag')
    learnRate = cfg.read('learningRate')
    data_cfg = cfg.read('dataConfig')
    fold_ratio = cfg.read('foldRatio')
    schFlag = cfg.read('schedulerFlag')
    schGamma = cfg.read('schedulerGamma')
    schMiles = cfg.read('schedulerMilestones')
    trainNo = cfg.read('trainNo')
    cfg.write("trainNo", trainNo+1)
    max_val_acc_global = cfg.read('maxValidationAcc')
    val_acc_thr = cfg.read('validationAccThr')
    log_batch_at = cfg.read('logBatchAt')
    weight_decay = cfg.read('weightDecay')
    dropout = cfg.read('dropoutRate')
    multistage_models = cfg.read('multiStageModelList')
    initUniform = cfg.read('lastLayerInitUniform')
    warm_cfg = cfg.read('warmStartConfig')
    model_version = cfg.read('modelVersion')
    dir_path = cfg.read('dataPath')
    fps = cfg.read('fps')

    # Set Devices
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('I - Running on device: {}'.format(device))
    print("I -", end=" ")
    ai8x.set_device(device=85, simulate=False, round_avg=False)

    # Set Dataloaders
    blacklist = blk.readAll()
    blacklist = [item for sublist in list(blacklist.values()) for item in sublist] # Flatten blacklist (that is dict of lists, initially)
    transform = transforms.Compose([transforms.ToTensor(), ai8x.normalize(args=config.Args(act_mode_8bit=False))])
    train_set = Kinetics(dir_path=dir_path, split='train', img_size=img_size, num_classes=num_classes, 
                fold_ratio=fold_ratio, fps=fps, num_frames_model=num_frames_model, num_frames_dataset=num_frames_dataset, transform=transform,
                data_configuration = data_cfg, blacklist = blacklist)
    if data_cfg['singleBackgroundPath'] and 'background' in label_names: train_set.load_next_background_pickle()
    print("I - Train set length: ", str(len(train_set)))
    print("I - Label distribution:", train_set.label_distribution)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = Kinetics(dir_path=dir_path, split='test', img_size=img_size, num_classes=num_classes, 
                fold_ratio=fold_ratio, fps=fps, num_frames_model=num_frames_model, num_frames_dataset=num_frames_dataset, transform=transform,
                data_configuration = data_cfg)
    if data_cfg['singleBackgroundPath'] and 'background' in label_names: test_set.load_next_background_pickle()
    print("I - Test set length: ", str(len(test_set)))
    print("I - Label distribution:", test_set.label_distribution)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Check Dataloader Output (run with a single batch)
    for data, label, ind in train_loader:
        print("I - Batch size: ", len(data), " tensor shape: ", data[0].shape," data min-max: ", data.min(), data.max())
        print("I - Label min-max: ", label.min(), label.max(), "data number in dataset: ", ind)
        break

    # Call TCN Model
    input_size = (int(img_size[0]/fold_ratio), int(img_size[1]/fold_ratio))
    tcn_caller = getattr(tcn, f'TCNv{model_version}')
    model = tcn_caller(num_classes=num_classes, input_size=input_size, 
            num_channels=3*fold_ratio*fold_ratio, num_frames_model=num_frames_model, bias=bias, dropout=dropout, initUniform=initUniform).to(device)
    softmax = torch.nn.Softmax(dim=1) # Used for histograms
    print(f'I - Number of Model Parameters: {model.count_parameters()}')

    # Warm Start
    if warm_cfg['warmStartFlag']:
        print("I - Warm start initiated")
        chck_model_version = warm_cfg['checkpointModelNo']
        tcn_caller = getattr(tcn, f'TCNv{chck_model_version}')
        checkpoint_model = tcn_caller(num_classes=num_classes, input_size=input_size,
                            num_channels=3*fold_ratio*fold_ratio, num_frames_model=num_frames_model, bias=bias, dropout=dropout, initUniform=initUniform).to(device)
        checkpoint_model, _, _, _ = apputils.load_checkpoint(model=checkpoint_model, model_device='cuda', chkpt_file=warm_cfg['checkpointFile'])
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model.state_dict(), strict=False) # Load state dictionary to the current model, and returns uncommon parameters
        print("I - Warm Start: Missing Keys", missing_keys)
        print("I - Warm Start: Unexpected Keys", unexpected_keys)
        del checkpoint_model # Delete loaded checkpoint model
        if warm_cfg['freezeSpatialCNN']: # Freeze common parameters of two models
            print(f'I - Freezing common parameters of models {model_version} and {chck_model_version}')
            for param in model.named_parameters():
                if param[0] not in missing_keys:
                    param[1].requires_grad = False

    # Check Model (with the data batch loaded during dataloader check)
    with torch.no_grad():
        model_out = model(data.to(device))
        print("I - Model output shape: ", model_out.shape)
        print("I - Model summary")
        summary(model, input_size=data.shape)

    # Prepare Training
    train_pred_dist = np.ones((num_classes, len(train_set)))*-1 # Tensorboard related, histogram arrays initialized with -1
    train_ground_dist = np.ones((num_classes, len(train_set)))*-1 # Tensorboard related, histogram arrays initialized with -1
    test_pred_dist = np.ones((num_classes, len(test_set)))*-1 # Tensorboard related, histogram arrays initialized with -1
    test_ground_dist = np.ones((num_classes, len(test_set)))*-1 # Tensorboard related, histogram arrays initialized with -1
    optimizer = torch.optim.Adam(model.parameters(), lr=learnRate, weight_decay=weight_decay, amsgrad=True)
    if schFlag:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schMiles, gamma=schGamma)

    if cls_weights_flag:
        cls_weightsT = torch.Tensor(cls_weights).to(device)
        loss_func = torch.nn.CrossEntropyLoss(weight=cls_weightsT).to(device)
    else:
        loss_func = torch.nn.CrossEntropyLoss().to(device)

    # Training & Evalutaion
    writer = SummaryWriter(comment=f'train_no_{trainNo}') # Tensorboard writer
    np.set_printoptions(suppress=True) # To suppress 'ndarray' and 'dtype' information while printing confusion matrix etc.
    max_val_acc_local = 0
    for epoch in range(num_epochs):
        print("I - Epoch: " + str(epoch))
        model.train()
        train_loss, correct, total, correctW, totalW = 0, 0, 0, 0, 0
        confusion_matrix_train = torch.zeros((num_classes, num_classes))

        # Training
        train_start = time.time()
        print("I - Training: ")
        for num_batch, (batch, labels, ind) in enumerate(train_loader, 1):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            if model_version in multistage_models: # For MS-TCN models losses are reduced by sum 
                loss = 0
                for stage_output in outputs:  
                    x = stage_output.transpose(2, 1).reshape(-1, num_classes) # [batch*frames, classes]
                    y = labels.repeat_interleave(num_frames_model) # Repeat labels, as all the frames have the same label. [batch*frames, 1]
                    loss += loss_func(x, y) # Sum for all frames of all samples in the batch
                outputs = torch.sum(outputs[-1, :, :, :], dim=2) # Take last stage output and reduce for accuracy calculation
            else:
                loss = loss_func(outputs, labels) # Standard loss function for non-multi-stage models
            loss.backward()
            optimizer.step()

            # Update Metrics & Confusion Matrix
            train_loss += loss.item()
            curr_total, curr_correct = compute_acc(outputs,labels)
            curr_totalW, curr_correctW = compute_weighted_acc(outputs, labels, cls_weights)
            total += curr_total
            correct += curr_correct
            totalW += curr_totalW
            correctW += curr_correctW
            for pred, lab in zip(outputs.max(1).indices, labels):
                    confusion_matrix_train[pred, lab] += 1
            prediction_dist(softmax(outputs), labels, num_batch-1, train_ground_dist, train_pred_dist, batch_size) # Update histogram arrays

            # Print Intermediate Results
            if (num_batch % log_batch_at) == 0:
                print(f'\tI - Batch: {num_batch} | Loss: {(train_loss / num_batch):.3f} | Acc: {(100 * correct / total):.3f}% | Wgt Acc: {(100 * correctW / totalW):.3f}%')
        
        # Print Metrics & Confusion Matrix
        write_histogram("Train", writer, epoch, train_ground_dist, train_pred_dist) # Write train histograms to tensorboard
        print("I - num batch:", num_batch)
        train_loss /= num_batch
        acc = 100.*(correct)/total
        accW = 100.*(correctW)/totalW
        error_rate = 100.*(total-correct)/total
        lr = optimizer.param_groups[0]['lr']
        train_end = time.time()
        print('I - Train -- Loss: %.3f | Acc: %.3f%% | Wgt Acc: %.3f%% | LR: %e | Dur: %.2fs' % (train_loss, acc, accW, lr, (train_end-train_start)))
        print('I - Confusion Matrix: [row->prediction - col->label]')
        print(confusion_matrix_train.numpy())
        print('')        

        # Validation
        print("I - Validation: ")
        model.eval()
        with torch.no_grad():
            val_loss, val_corr, val_tot, val_corrW, val_totW = 0, 0, 0, 0, 0
            confusion_matrix_valid = torch.zeros((num_classes, num_classes))
            train_start = time.time()
            val_results = np.zeros((len(test_set), 4))
            for num_batch, (batch, labels, ind) in enumerate(test_loader, 1):
                batch, labels = batch.to(device), labels.to(device)
                outputs = model(batch)
                if model_version in multistage_models: # For MS-TCN models losses are reduced by sum 
                    loss = 0
                    for stage_output in outputs:
                        x = stage_output.transpose(2, 1).reshape(-1, num_classes) # [batch*frames, classes]
                        y = labels.repeat(num_frames_model) # Repeat labels, as all the frames have the same label. [batch*frames, 1]
                        loss += loss_func(x, y) # Sum for all frames of all samples in the batch
                    outputs = torch.sum(outputs[-1, :, :, :], dim=2) # Take last stage output and reduce for accuracy calculation
                else:
                    loss = loss_func(outputs, labels) # Standard loss function for non-multi-stage models

                # Update Metrics & Confusion Matrix
                val_loss += loss.item()
                curr_total, curr_correct = compute_acc(outputs,labels)
                curr_totalW, curr_correctW = compute_weighted_acc(outputs, labels, cls_weights)
                val_tot += curr_total
                val_corr += curr_correct
                val_totW += curr_totalW
                val_corrW += curr_correctW               
                for n, (pred, lab) in enumerate(zip(outputs.max(1).indices, labels)):
                    confusion_matrix_valid[pred, lab] += 1
                    val_results[(num_batch-1)*batch_size+n, 0] = ind[n] # Validation results to be printed in case new maximum value is reached
                    val_results[(num_batch-1)*batch_size+n, 1] = lab
                    val_results[(num_batch-1)*batch_size+n, 2] = pred
                    val_results[(num_batch-1)*batch_size+n, 3] = outputs.max(1).values[n]                     
                prediction_dist(softmax(outputs), labels, num_batch-1, test_ground_dist, test_pred_dist, batch_size) # Update histogram arrays

                # Print Intermediate Results
                if (num_batch % log_batch_at) == 0:
                    print(f'\tI - Batch: {num_batch} | Loss: {(val_loss / num_batch):.3f} | Acc: {(100 * val_corr / val_tot):.3f}% | Wgt Acc: {(100 * val_corrW / val_totW):.3f}%')
            
            # Print Metrics & Confusion Matrix
            write_histogram("Test", writer, epoch, test_ground_dist, test_pred_dist) # Write train histograms to tensorboard
            print("I - num batch:", num_batch)
            val_loss /= num_batch
            val_acc = 100.*(val_corr)/val_tot
            val_accW = 100.*(val_corrW)/val_totW
            train_end = time.time()
            print('I - Val -- Loss: %.3f | Acc: %.3f%% | Wgt Acc: %.3f%% | Dur: %.2fs' % (val_loss, val_acc, val_accW, (train_end-train_start)))
            print('I - Confusion Matrix: [row->prediction - col->label]')
            print(confusion_matrix_valid.numpy())
            print('')

            # Check Current Training Best Result
            if val_acc > max_val_acc_local:
                max_val_acc_local = val_acc
                print("I - Local maximum validation set accuracy: ", f'{max_val_acc_local:.2f}\n')                
                print("I - Validation set results: ")
                print_neat(val_results)
                print("\n---------------------------")
                # Check Global Best or Surpassed Manual Threshold 
                if (max_val_acc_local > max_val_acc_global) or (max_val_acc_local > val_acc_thr):
                    max_val_acc_global = max_val_acc_local
                    print("I - Global or recent maximum validation set accuracy: ", f'{max_val_acc_global:.2f}')
                    cfg.write("maxValidationAcc", max_val_acc_global)
                    cfg.write("maxValidationTrainNo", trainNo)
                    
                    # Save Checkpoint
                    arch_name = f'noQAT'
                    model_name = f'model{model_version}_trainNo{trainNo}_at_epoch_{epoch}_with_acc_{max_val_acc_global:.2f}'.replace('.','_')
                    apputils.save_checkpoint(epoch, arch_name, model, optimizer=optimizer,
                                            scheduler=None, extras=None,
                                            is_best=False, name=model_name, dir='./sav')
        
        if data_cfg['singleBackgroundPath'] and 'background' in label_names: train_set.load_next_background_pickle() # Load next background pkl (single back pkl mode)                                         
        # Update Tensorboard Scalars
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)
        writer.add_scalar('Accuracy/train(weighted)', accW, epoch)
        writer.add_scalar('Accuracy/test', val_acc, epoch)
        writer.add_scalar('Accuracy/test(weighted)', val_accW, epoch)
        if schFlag:
            scheduler.step()
    writer.close()
    print("I - Maximum validation set accuracy in current training: ", f'{max_val_acc_local:.2f}')