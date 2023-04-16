import sys
sys.path.append('../tcn')
import torch
from tcn import TCNv9
import ai8x
from distiller import apputils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ai8x.set_device(device=85, simulate=False, round_avg=False)

input_size = (64, 64)
model_in = TCNv9(num_classes=4, input_size=input_size, num_channels=48, num_frames_model=16, bias=True, dropout=0.5, initUniform=True).to(device)
model_out, sch, opt, epoch = apputils.load_checkpoint(model=model_in, model_device='cuda', chkpt_file='./sav/model9_trainNo39_at_epoch_22_with_acc_70_34_checkpoint.pth.tar')