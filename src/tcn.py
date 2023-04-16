###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# Temporal Convolutional Network models and simpler CNN models for video understanding & classification.

import ai8x
import torch

from torch import nn

class TCNBase(nn.Module): # Base class for TCN models
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.input_size = input_size
        self.num_frames_model = num_frames_model
        self.bias = bias
        self.bn = 'Affine'
        self.cnn_out_shape = (0, 0)
        self.cnn_out_channel = 0
        self.p = dropout
        print("I - Initializing model "+self.__class__.__name__)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TCNv1(TCNBase): # Complex CNN (14 conv) w/ TCN (3 stages in parallel) & Linear (2 layers)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)
        cnn_down_factor = 32
        nfil = 64
        self.cnn_out_channel = 64
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        len_tcn_output = (num_frames_model - 14)*nfil # Due to temporal operations of TCN
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.parp1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
                
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1d_1 = ai8x.FusedConv1dBNReLU(len_frame_vector, nfil, 5, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 5, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 5, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.conv1d_p2_1 = ai8x.FusedConv1dBNReLU(len_frame_vector,  nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p2_2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p2_3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p2_4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1d_p3_1 = ai8x.FusedConv1dBNReLU(len_frame_vector,  nfil, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p3_2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p3_3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=4, bias=bias, batchnorm=self.bn, **kwargs)

        self.fc1 = ai8x.FusedLinearReLU(len_tcn_output, 32, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(32, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        c = self.conv1(x)
        c = self.conv2(c)
        cp = self.parp1(x)
        c = c + cp
        c = self.conv3(c)
        c = self.conv4(c)
        c = self.conv5(c)
        cx = self.conv6(c)
        
        c = self.conv7(cx)
        c = self.conv7_1(c)
        cp = self.conv7_p(cx)
        cx = cp + c

        c = self.conv8(cx)
        c = self.conv8_1(c)
        cp = self.conv8_p(cx)
        cx = cp + c

        c = self.conv9(cx)
        c = self.conv9_1(c)
        cp = self.conv9_p(cx)
        c = cp + c
        return c

    def create_tcn(self, x):
        c = self.conv1d_1(x)
        c = self.conv1d_2(c)
        c = self.conv1d_3(c)
        c = self.conv1d_4(c)
        return c
    
    def create_tcn_p2(self, x):
        c = self.conv1d_p2_1(x)
        c = self.conv1d_p2_2(c)
        c = self.conv1d_p2_3(c)
        c = self.conv1d_p2_4(c)
        return c

    def create_tcn_p3(self, x):
        c = self.conv1d_p3_1(x)
        c = self.conv1d_p3_2(c)
        c = self.conv1d_p3_3(c)
        return c

    def create_dense(self, x):
        c = self.fc1(x)
        c = self.fc2(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        cnnoutputs = cnnoutputs.permute(0, 1, 3, 4, 2)
        cnn_sh = cnnoutputs.shape
        tcninput = cnnoutputs.reshape(cnn_sh[0], cnn_sh[1], -1) #might use reshape
        tcninput = tcninput.permute(0, 2, 1)

        #Call TCN
        tcnoutput = self.create_tcn(tcninput)
        tcnoutput_p2 = self.create_tcn_p2(tcninput)
        tcnoutput_p3 = self.create_tcn_p3(tcninput)
        
        #Aggregation
        tcnoutput = tcnoutput + tcnoutput_p2 + tcnoutput_p3

        # Flatten
        classifierinput = tcnoutput.view(tcnoutput.size(0), -1)

        # Call dense layers
        outputs = self.create_dense(classifierinput)

        return outputs

class TCNv2(TCNBase): # Complex CNN (14 conv) w/ TCN (4 stages in parallel) & Linear (2 layers)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)
        cnn_down_factor = 32
        nfil = 64
        self.cnn_out_channel = 64
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        len_tcn_output = (num_frames_model - 14)*nfil # Due to temporal operations of TCN
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.parp1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
                
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1d_1 = ai8x.FusedConv1dBNReLU(len_frame_vector, nfil, 5, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 5, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 5, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.conv1d_p2_1 = ai8x.FusedConv1dBNReLU(len_frame_vector,  nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p2_2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p2_3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p2_4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1d_p3_1 = ai8x.FusedConv1dBNReLU(len_frame_vector,  nfil, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p3_2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p3_3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=4, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1d_p4_1 = ai8x.FusedConv1dBNReLU(len_frame_vector,  nfil, 9, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p4_2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 5, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1d_p4_3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.fc1 = ai8x.FusedLinearReLU(len_tcn_output, 32, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(32, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        c = self.conv1(x)
        c = self.conv2(c)
        cp = self.parp1(x)
        c = c + cp
        c = self.conv3(c)
        c = self.conv4(c)
        c = self.conv5(c)
        cx = self.conv6(c)
        
        c = self.conv7(cx)
        c = self.conv7_1(c)
        cp = self.conv7_p(cx)
        cx = cp + c

        c = self.conv8(cx)
        c = self.conv8_1(c)
        cp = self.conv8_p(cx)
        cx = cp + c

        c = self.conv9(cx)
        c = self.conv9_1(c)
        cp = self.conv9_p(cx)
        c = cp + c
        return c

    def create_tcn(self, x):
        c = self.conv1d_1(x)
        c = self.conv1d_2(c)
        c = self.conv1d_3(c)
        c = self.conv1d_4(c)
        return c
    
    def create_tcn_p2(self, x):
        c = self.conv1d_p2_1(x)
        c = self.conv1d_p2_2(c)
        c = self.conv1d_p2_3(c)
        c = self.conv1d_p2_4(c)
        return c

    def create_tcn_p3(self, x):
        c = self.conv1d_p3_1(x)
        c = self.conv1d_p3_2(c)
        c = self.conv1d_p3_3(c)
        return c

    def create_tcn_p4(self, x):
        c = self.conv1d_p4_1(x)
        c = self.conv1d_p4_2(c)
        c = self.conv1d_p4_3(c)
        return c
        
    def create_dense(self, x):
        c = self.fc1(x)
        c = self.fc2(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        cnnoutputs = cnnoutputs.permute(0, 1, 3, 4, 2)
        cnn_sh = cnnoutputs.shape
        tcninput = cnnoutputs.reshape(cnn_sh[0], cnn_sh[1], -1) #might use reshape
        tcninput = tcninput.permute(0, 2, 1)

        #Call TCN
        tcnoutput = self.create_tcn(tcninput)
        tcnoutput_p2 = self.create_tcn_p2(tcninput)
        tcnoutput_p3 = self.create_tcn_p3(tcninput)
        tcnoutput_p4 = self.create_tcn_p4(tcninput)

        #Aggregation
        tcnoutput = tcnoutput + tcnoutput_p2 + tcnoutput_p3 + tcnoutput_p4 

        # Flatten
        classifierinput = tcnoutput.view(tcnoutput.size(0), -1)

        # Call dense layers
        outputs = self.create_dense(classifierinput)

        return outputs

class TCNv3(TCNBase): # Complex CNN (14 conv) w/ TCN (4 stages in parallel) & Linear (2 layers)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)
        cnn_down_factor = 32
        nfil = 64
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel*num_frames_model
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.parp1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
                
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.fc1 = ai8x.FusedLinearReLU(len_frame_vector, 32, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(32, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        c = self.conv1(x)
        c = self.conv2(c)
        cp = self.parp1(x)
        c = c + cp
        c = self.conv3(c)
        c = self.conv4(c)
        c = self.conv5(c)
        cx = self.conv6(c)
        
        c = self.conv7(cx)
        c = self.conv7_1(c)
        cp = self.conv7_p(cx)
        cx = cp + c

        c = self.conv8(cx)
        c = self.conv8_1(c)
        cp = self.conv8_p(cx)
        cx = cp + c

        c = self.conv9(cx)
        c = self.conv9_1(c)
        cp = self.conv9_p(cx)
        c = cp + c
        return c
     
    def create_dense(self, x):
        c = self.fc1(x)
        c = self.fc2(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        cnnoutputs = cnnoutputs.permute(0, 3, 4, 2, 1) # [batch, H, W, nfil, #frames]
        classifierinput = cnnoutputs.reshape(cnnoutputs.shape[0], -1)

        # Call dense layers
        outputs = self.create_dense(classifierinput)

        return outputs

class TCNv4(TCNBase): # Complex CNN (15 conv, linear) w/ Soft voting
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 32
        nfil = 64
        self.cnn_out_channel = 64
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c

        c = self.conv2(cx)
        c = self.conv3(c)
        c = self.conv4(c)
        c = self.conv5(c)
        cx = self.conv6(c)
        
        c = self.conv7(cx)
        c = self.conv7_1(c)
        cp = self.conv7_p(cx)
        cx = cp + c

        c = self.conv8(cx)
        c = self.conv8_1(c)
        cp = self.conv8_p(cx)
        cx = cp + c

        c = self.conv9(cx)
        c = self.conv9_1(c)
        cp = self.conv9_p(cx)
        c = cp + c
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs

class TCNv5(TCNBase): # Complex CNN (14 conv) w/ SS-TCN (1 stage) & Linear (2 layers)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)
        cnn_down_factor = 32
        nfil = 64
        self.cnn_out_channel = 64
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        len_tcn_output = (num_frames_model-8)*nfil # Due to temporal operations of TCN
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.parp1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
                
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.tcn_s0 = ai8x.Conv1d(len_frame_vector, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.tcn_s1_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=4, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=4, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        
        self.fc1 = ai8x.FusedLinearReLU(len_tcn_output, 32, bias=True, **kwargs)
        self.fc2 = ai8x.Linear(32, num_classes, wide=True, bias=False, **kwargs)

        if kw['initUniform']:
            nn.init.constant_(self.fc2.op.weight, 0.25)            

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        c = self.conv1(x)
        c = self.conv2(c)
        cp = self.parp1(x)
        c = c + cp
        c = self.conv3(c)
        c = self.conv4(c)
        c = self.conv5(c)
        cx = self.conv6(c)
        
        c = self.conv7(cx)
        c = self.conv7_1(c)
        cp = self.conv7_p(cx)
        cx = cp + c

        c = self.conv8(cx)
        c = self.conv8_1(c)
        cp = self.conv8_p(cx)
        cx = cp + c

        c = self.conv9(cx)
        c = self.conv9_1(c)
        cp = self.conv9_p(cx)
        c = cp + c
        return c

    def create_tcn(self, x):
        c = self.tcn_s1_l1(x)
        c = self.tcn_s1_l1_1x1(c)
        cp = c + x
        c = self.tcn_s1_l2(cp)
        c = self.tcn_s1_l2_1x1(c)
        cp = c + cp
        c = self.tcn_s1_l3(cp)
        c = self.tcn_s1_l3_1x1(c)
        cp = c + cp[:,:,2:-2]
        c = self.tcn_s1_l4(cp)
        c = self.tcn_s1_l4_1x1(c)
        cp = c + cp[:,:,2:-2]
        return cp
        
    def create_dense(self, x):
        c = self.fc1(x)
        c = self.fc2(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        cnnoutputs = cnnoutputs.permute(0, 1, 3, 4, 2)
        cnn_sh = cnnoutputs.shape
        tcninput = cnnoutputs.reshape(cnn_sh[0], cnn_sh[1], -1) #might use reshape
        tcninput = tcninput.permute(0, 2, 1)

        #Call TCN
        tcninput_correct_chn = self.tcn_s0(tcninput)
        tcnoutput = self.create_tcn(tcninput_correct_chn)

        # Flatten
        classifierinput = tcnoutput.view(tcnoutput.size(0), -1)

        # Call dense layers
        outputs = self.create_dense(classifierinput)

        return outputs

class TCNv6(TCNBase): # Complex CNN (14 conv) w/ MS-TCN (3 stages)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)
        cnn_down_factor = 32
        nfil = 64
        self.cnn_out_channel = 64
        kwargs = {}
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.parp1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
                
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.tcn_s1_in = ai8x.Conv1d(len_frame_vector, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s1_l1 = nn.Dropout(self.p)
        self.drp_s1_l2 = nn.Dropout(self.p)
        self.drp_s1_l3 = nn.Dropout(self.p)
        self.drp_s1_l4 = nn.Dropout(self.p)
        self.tcn_s1_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.tcn_s2_in = ai8x.Conv1d(num_classes, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s2_l1 = nn.Dropout(self.p)
        self.drp_s2_l2 = nn.Dropout(self.p)
        self.drp_s2_l3 = nn.Dropout(self.p)
        self.drp_s2_l4 = nn.Dropout(self.p)
        self.tcn_s2_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.tcn_s3_in = ai8x.Conv1d(num_classes, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s3_l1 = nn.Dropout(self.p)
        self.drp_s3_l2 = nn.Dropout(self.p)
        self.drp_s3_l3 = nn.Dropout(self.p)
        self.drp_s3_l4 = nn.Dropout(self.p)
        self.tcn_s3_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        if kw['initUniform']:
            nn.init.zeros_(self.tcn_s1_out.op.weight)
            nn.init.zeros_(self.tcn_s2_out.op.weight)
            nn.init.zeros_(self.tcn_s3_out.op.weight)


    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        c = self.conv1(x)
        c = self.conv2(c)
        cp = self.parp1(x)
        c = c + cp
        c = self.conv3(c)
        c = self.conv4(c)
        c = self.conv5(c)
        cx = self.conv6(c)
        
        c = self.conv7(cx)
        c = self.conv7_1(c)
        cp = self.conv7_p(cx)
        cx = cp + c

        c = self.conv8(cx)
        c = self.conv8_1(c)
        cp = self.conv8_p(cx)
        cx = cp + c

        c = self.conv9(cx)
        c = self.conv9_1(c)
        cp = self.conv9_p(cx)
        c = cp + c
        return c

    def tcn_stage1(self, x) -> torch.Tensor:
        cp = self.tcn_s1_in(x)
        c = self.tcn_s1_l1(cp)
        c = self.tcn_s1_l1_1x1(c)
        c = self.drp_s1_l1(c)
        cp = c + cp
        c = self.tcn_s1_l2(cp)
        c = self.tcn_s1_l2_1x1(c)
        c = self.drp_s1_l2(c)
        cp = c + cp
        c = self.tcn_s1_l3(cp)
        c = self.tcn_s1_l3_1x1(c)
        c = self.drp_s1_l3(c)
        cp = c + cp
        c = self.tcn_s1_l4(cp)
        c = self.tcn_s1_l4_1x1(c)
        c = self.drp_s1_l4(c)
        cp = c + cp
        cp = self.tcn_s1_out(cp)
        return cp

    def tcn_stage2(self, x) -> torch.Tensor:
        cp = self.tcn_s2_in(x)
        c = self.tcn_s2_l1(cp)
        c = self.tcn_s2_l1_1x1(c)
        c = self.drp_s2_l1(c)
        cp = c + cp
        c = self.tcn_s2_l2(cp)
        c = self.tcn_s2_l2_1x1(c)
        c = self.drp_s2_l2(c)
        cp = c + cp
        c = self.tcn_s2_l3(cp)
        c = self.tcn_s2_l3_1x1(c)
        c = self.drp_s2_l3(c)
        cp = c + cp
        c = self.tcn_s2_l4(cp)
        c = self.tcn_s2_l4_1x1(c)
        c = self.drp_s2_l4(c)
        cp = c + cp
        cp = self.tcn_s2_out(cp)
        return cp

    def tcn_stage3(self, x) -> torch.Tensor:
        cp = self.tcn_s3_in(x)
        c = self.tcn_s3_l1(cp)
        c = self.tcn_s3_l1_1x1(c)
        c = self.drp_s3_l1(c)
        cp = c + cp
        c = self.tcn_s3_l2(cp)
        c = self.tcn_s3_l2_1x1(c)
        c = self.drp_s3_l2(c)
        cp = c + cp
        c = self.tcn_s3_l3(cp)
        c = self.tcn_s3_l3_1x1(c)
        c = self.drp_s3_l3(c)
        cp = c + cp
        c = self.tcn_s3_l4(cp)
        c = self.tcn_s3_l4_1x1(c)
        c = self.drp_s3_l4(c)
        cp = c + cp
        cp = self.tcn_s3_out(cp)
        return cp
        
    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        cnnoutputs = cnnoutputs.permute(0, 1, 3, 4, 2)
        cnn_sh = cnnoutputs.shape
        tcninput = cnnoutputs.reshape(cnn_sh[0], cnn_sh[1], -1) #might use reshape
        tcninput = tcninput.permute(0, 2, 1)

        #TCN stages
        tcnoutput = self.tcn_stage1(tcninput)
        outputs = tcnoutput.unsqueeze(0)
        tcnoutput = self.tcn_stage2(tcnoutput)
        outputs = torch.cat((outputs, tcnoutput.unsqueeze(0)), dim=0)
        tcnoutput = self.tcn_stage3(tcnoutput)
        outputs = torch.cat((outputs, tcnoutput.unsqueeze(0)), dim=0)

        return outputs

class TCNv7(TCNBase): # Complex CNN (14 conv) w/ MS-TCN (5 stages)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)
        cnn_down_factor = 32
        nfil = 64
        self.cnn_out_channel = 64
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv7_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv8_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv9_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)

        self.tcn_s1_in = ai8x.Conv1d(len_frame_vector, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s1_l1 = nn.Dropout(self.p)
        self.drp_s1_l2 = nn.Dropout(self.p)
        self.drp_s1_l3 = nn.Dropout(self.p)
        self.drp_s1_l4 = nn.Dropout(self.p)
        self.tcn_s1_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.tcn_s2_in = ai8x.Conv1d(num_classes, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s2_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s2_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s2_l1 = nn.Dropout(self.p)
        self.drp_s2_l2 = nn.Dropout(self.p)
        self.drp_s2_l3 = nn.Dropout(self.p)
        self.drp_s2_l4 = nn.Dropout(self.p)
        self.tcn_s2_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.tcn_s3_in = ai8x.Conv1d(num_classes, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s3_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s3_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s3_l1 = nn.Dropout(self.p)
        self.drp_s3_l2 = nn.Dropout(self.p)
        self.drp_s3_l3 = nn.Dropout(self.p)
        self.drp_s3_l4 = nn.Dropout(self.p)
        self.tcn_s3_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.tcn_s4_in = ai8x.Conv1d(num_classes, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s4_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s4_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s4_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s4_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s4_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s4_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s4_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s4_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s4_l1 = nn.Dropout(self.p)
        self.drp_s4_l2 = nn.Dropout(self.p)
        self.drp_s4_l3 = nn.Dropout(self.p)
        self.drp_s4_l4 = nn.Dropout(self.p)
        self.tcn_s4_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.tcn_s5_in = ai8x.Conv1d(num_classes, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s5_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s5_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s5_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s5_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s5_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s5_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s5_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s5_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s5_l1 = nn.Dropout(self.p)
        self.drp_s5_l2 = nn.Dropout(self.p)
        self.drp_s5_l3 = nn.Dropout(self.p)
        self.drp_s5_l4 = nn.Dropout(self.p)
        self.tcn_s5_out = ai8x.Conv1d(nfil, num_classes, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)

        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c

        c = self.conv2(cx)
        c = self.conv3(c)
        c = self.conv4(c)
        c = self.conv5(c)
        cx = self.conv6(c)
        
        c = self.conv7(cx)
        c = self.conv7_1(c)
        cp = self.conv7_p(cx)
        cx = cp + c

        c = self.conv8(cx)
        c = self.conv8_1(c)
        cp = self.conv8_p(cx)
        cx = cp + c

        c = self.conv9(cx)
        c = self.conv9_1(c)
        cp = self.conv9_p(cx)
        c = cp + c
        return c

    def tcn_stage1(self, x) -> torch.Tensor:
        cp = self.tcn_s1_in(x)
        c = self.tcn_s1_l1(cp)
        c = self.tcn_s1_l1_1x1(c)
        c = self.drp_s1_l1(c)
        cp = c + cp
        c = self.tcn_s1_l2(cp)
        c = self.tcn_s1_l2_1x1(c)
        c = self.drp_s1_l2(c)
        cp = c + cp
        c = self.tcn_s1_l3(cp)
        c = self.tcn_s1_l3_1x1(c)
        c = self.drp_s1_l3(c)
        cp = c + cp
        c = self.tcn_s1_l4(cp)
        c = self.tcn_s1_l4_1x1(c)
        c = self.drp_s1_l4(c)
        cp = c + cp
        cp = self.tcn_s1_out(cp)
        return cp

    def tcn_stage2(self, x) -> torch.Tensor:
        cp = self.tcn_s2_in(x)
        c = self.tcn_s2_l1(cp)
        c = self.tcn_s2_l1_1x1(c)
        c = self.drp_s2_l1(c)
        cp = c + cp
        c = self.tcn_s2_l2(cp)
        c = self.tcn_s2_l2_1x1(c)
        c = self.drp_s2_l2(c)
        cp = c + cp
        c = self.tcn_s2_l3(cp)
        c = self.tcn_s2_l3_1x1(c)
        c = self.drp_s2_l3(c)
        cp = c + cp
        c = self.tcn_s2_l4(cp)
        c = self.tcn_s2_l4_1x1(c)
        c = self.drp_s2_l4(c)
        cp = c + cp
        cp = self.tcn_s2_out(cp)
        return cp

    def tcn_stage3(self, x) -> torch.Tensor:
        cp = self.tcn_s3_in(x)
        c = self.tcn_s3_l1(cp)
        c = self.tcn_s3_l1_1x1(c)
        c = self.drp_s3_l1(c)
        cp = c + cp
        c = self.tcn_s3_l2(cp)
        c = self.tcn_s3_l2_1x1(c)
        c = self.drp_s3_l2(c)
        cp = c + cp
        c = self.tcn_s3_l3(cp)
        c = self.tcn_s3_l3_1x1(c)
        c = self.drp_s3_l3(c)
        cp = c + cp
        c = self.tcn_s3_l4(cp)
        c = self.tcn_s3_l4_1x1(c)
        c = self.drp_s3_l4(c)
        cp = c + cp
        cp = self.tcn_s3_out(cp)
        return cp

    def tcn_stage4(self, x) -> torch.Tensor:
        cp = self.tcn_s4_in(x)
        c = self.tcn_s4_l1(cp)
        c = self.tcn_s4_l1_1x1(c)
        c = self.drp_s4_l1(c)
        cp = c + cp
        c = self.tcn_s4_l2(cp)
        c = self.tcn_s4_l2_1x1(c)
        c = self.drp_s4_l2(c)
        cp = c + cp
        c = self.tcn_s4_l3(cp)
        c = self.tcn_s4_l3_1x1(c)
        c = self.drp_s4_l3(c)
        cp = c + cp
        c = self.tcn_s4_l4(cp)
        c = self.tcn_s4_l4_1x1(c)
        c = self.drp_s4_l4(c)
        cp = c + cp
        cp = self.tcn_s4_out(cp)
        return cp

    def tcn_stage5(self, x) -> torch.Tensor:
        cp = self.tcn_s5_in(x)
        c = self.tcn_s5_l1(cp)
        c = self.tcn_s5_l1_1x1(c)
        c = self.drp_s5_l1(c)
        cp = c + cp
        c = self.tcn_s5_l2(cp)
        c = self.tcn_s5_l2_1x1(c)
        c = self.drp_s5_l2(c)
        cp = c + cp
        c = self.tcn_s5_l3(cp)
        c = self.tcn_s5_l3_1x1(c)
        c = self.drp_s5_l3(c)
        cp = c + cp
        c = self.tcn_s5_l4(cp)
        c = self.tcn_s5_l4_1x1(c)
        c = self.drp_s5_l4(c)
        cp = c + cp
        cp = self.tcn_s5_out(cp)
        return cp
        
    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        cnnoutputs = cnnoutputs.permute(0, 1, 3, 4, 2)
        tcninput = cnnoutputs.reshape(batch_size, num_frames, -1)
        tcninput = tcninput.permute(0, 2, 1)

        #TCN stages
        tcnoutput = self.tcn_stage1(tcninput)
        outputs = tcnoutput.unsqueeze(0)
        tcnoutput = self.tcn_stage2(tcnoutput)
        outputs = torch.cat((outputs, tcnoutput.unsqueeze(0)), dim=0)
        tcnoutput = self.tcn_stage3(tcnoutput)
        outputs = torch.cat((outputs, tcnoutput.unsqueeze(0)), dim=0)
        tcnoutput = self.tcn_stage4(tcnoutput)
        outputs = torch.cat((outputs, tcnoutput.unsqueeze(0)), dim=0)
        tcnoutput = self.tcn_stage5(tcnoutput)
        outputs = torch.cat((outputs, tcnoutput.unsqueeze(0)), dim=0)

        return outputs

class TCNv8(TCNBase): # Simple CNN (10 conv, linear) w/ Soft voting
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 16
        nfil = 64
        self.cnn_out_channel = 64
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + c

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        cx = cp + c

        c = self.conv4(cx)
        c = self.conv4_1(c)
        cp = self.conv4_p(cx)
        c = cp + c
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs

class TCNv9(TCNBase): # Simple CNN (8 conv, linear) w/ Soft voting
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 64
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + c

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        c = cp + c
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs

class TCNv10(TCNBase): # Simple CNN (8 conv) w/ TCN (1 layer)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 64
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.tcn0 = ai8x.Conv1d(len_frame_vector, num_classes, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + c

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        cx = cp + c
        return cx

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        tcn_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1).permute(0, 2, 1)
        tcn_output = self.tcn0(tcn_input)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames-2):
            outputs += tcn_output[:, :, i] / (num_frames - 2)

        return outputs

class TCNv11(TCNBase): # Simple CNN (8 conv, linear) w/ Soft voting & Softmax
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 64
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)
        self.softmax = nn.Softmax()

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + c

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        c = cp + c
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.softmax(self.fc(linear_input[:, i])) / num_frames

        return outputs

class TCNv12(TCNBase): # Simple CNN (8 conv) w/ TCN (1 layer) & Linear (1 layer)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 64
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        len_tcn_output = num_frames_model*nfil # Due to temporal operations of TCN
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.tcn0 = ai8x.Conv1d(len_frame_vector, nfil, 5, padding=2, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.linear0 = ai8x.Linear(len_tcn_output, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + c

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        cx = cp + c
        return cx

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        tcn_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1).permute(0, 2, 1)
        tcn_output = self.tcn0(tcn_input)

        linear_input = tcn_output.permute(0, 2, 1).reshape(batch_size, -1)
        outputs = self.linear0(linear_input)

        # outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        # for i in range(num_frames-2):
        #     outputs += tcn_output[:, :, i] / (num_frames - 2)

        return outputs

class TCNv13(TCNBase): # Simple CNN (8 conv) w/ SS-TCN (1 stage) & Linear (2 layers)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 64
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        len_tcn_output = num_frames_model*nfil # Due to temporal operations of TCN
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.tcn_s1_in = ai8x.Conv1d(len_frame_vector, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l1 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=1, stride=1, dilation=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l2 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l3 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l4 = ai8x.FusedConv1dBNReLU(nfil, nfil, 3, padding=2, stride=1, dilation=2, bias=bias, batchnorm=self.bn, **kwargs)
        self.tcn_s1_l1_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l2_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l3_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.tcn_s1_l4_1x1 = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        self.drp_s1_l1 = nn.Dropout(self.p)
        self.drp_s1_l2 = nn.Dropout(self.p)
        self.drp_s1_l3 = nn.Dropout(self.p)
        self.drp_s1_l4 = nn.Dropout(self.p)
        self.tcn_s1_out = ai8x.Conv1d(nfil, nfil, 1, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)

        self.linear0 = ai8x.FusedLinearReLU(len_tcn_output, 32, bias=True, **kwargs)
        self.linear1 = ai8x.Linear(32, num_classes, wide=True, bias=False, **kwargs)

    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + c

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        cx = cp + c
        return cx

    def tcn_stage1(self, x) -> torch.Tensor:
        cp = self.tcn_s1_in(x)
        c = self.tcn_s1_l1(cp)
        c = self.tcn_s1_l1_1x1(c)
        c = self.drp_s1_l1(c)
        cp = c + cp
        c = self.tcn_s1_l2(cp)
        c = self.tcn_s1_l2_1x1(c)
        c = self.drp_s1_l2(c)
        cp = c + cp
        c = self.tcn_s1_l3(cp)
        c = self.tcn_s1_l3_1x1(c)
        c = self.drp_s1_l3(c)
        cp = c + cp
        c = self.tcn_s1_l4(cp)
        c = self.tcn_s1_l4_1x1(c)
        c = self.drp_s1_l4(c)
        cp = c + cp
        cp = self.tcn_s1_out(cp)
        return cp

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        tcn_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1).permute(0, 2, 1)
        tcn_output = self.tcn_stage1(tcn_input)

        linear_input = tcn_output.permute(0, 2, 1).reshape(batch_size, -1)
        outputs = self.linear1(self.linear0(linear_input))

        return outputs

class TCNv14(TCNBase): # Simple CNN (8 conv, linear, 128 filters) w/ Soft voting
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 128
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + c
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + c

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        c = cp + c
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs

class TCNv15(TCNBase): # Simple CNN (8 conv, linear, 128 filters, 3 dropouts) w/ Soft voting
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 128
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.drop1 = nn.Dropout2d(self.p)
        self.drop2 = nn.Dropout2d(self.p)
        self.drop3 = nn.Dropout2d(self.p)

        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + self.drop1(c)
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + self.drop2(c)

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        c = cp + self.drop3(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs

class TCNv16(TCNBase): # Simple CNN (8 conv, 128 filters, 3 dropouts) w/ TCN (1 layer)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 128
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.drop1 = nn.Dropout2d(self.p)
        self.drop2 = nn.Dropout2d(self.p)
        self.drop3 = nn.Dropout2d(self.p)

        self.tcn0 = ai8x.Conv1d(len_frame_vector, num_classes, 3, padding=0, stride=1, dilation=1, bias=bias, batchnorm=None, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + self.drop1(c)
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + self.drop2(c)

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        c = cp + self.drop3(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        tcn_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1).permute(0, 2, 1)
        tcn_output = self.tcn0(tcn_input)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames-2):
            outputs += tcn_output[:, :, i] / (num_frames - 2)

        return outputs

class TCNv17(TCNBase): # Simpler CNN (6 conv, linear, 128 filters, 3 dropouts) w/ Soft voting
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 128
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.drop0 = nn.Dropout2d(self.p)
        self.drop1 = nn.Dropout2d(self.p)
        self.drop2 = nn.Dropout2d(self.p)

        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        cx = self.drop0(cx)

        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + self.drop1(c)
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + self.drop2(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs

class TCNv18(TCNBase): # Simpler CNN (6 conv, 128 filters, 3 dropouts) w/ TCN (1 layer)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 128
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.drop0 = nn.Dropout2d(self.p)
        self.drop1 = nn.Dropout2d(self.p)
        self.drop2 = nn.Dropout2d(self.p)

        self.tcn0 = ai8x.Conv1d(len_frame_vector, num_classes, 5, padding=2, stride=1, dilation=1, bias=True, batchnorm=None, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        cx = self.drop0(cx)

        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + self.drop1(c)
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + self.drop2(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        tcn_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1).permute(0, 2, 1)
        tcn_output = self.tcn0(tcn_input)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += tcn_output[:, :, i] / (num_frames)

        return outputs

class TCNv19(TCNBase): # Simpler CNN (6 conv, 128 filters, 3 dropouts) w/ TCN (2 layers)
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 128
        self.cnn_out_channel = 16
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.drop0 = nn.Dropout2d(self.p)
        self.drop1 = nn.Dropout2d(self.p)
        self.drop2 = nn.Dropout2d(self.p)

        self.tcn0 = ai8x.Conv1d(len_frame_vector, nfil, 5, padding=0, stride=1, dilation=1, bias=True, batchnorm=None, **kwargs)
        self.tcn1 = ai8x.Conv1d(nfil, num_classes, 5, padding=0, stride=1, dilation=1, bias=True, batchnorm=None, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        cx = self.drop0(cx)

        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + self.drop1(c)
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + self.drop2(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        tcn_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1).permute(0, 2, 1)
        tcn_output = self.tcn1(self.tcn0(tcn_input))

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames - 8):
            outputs += tcn_output[:, :, i] / (num_frames - 8)

        return outputs

class TCNv20(TCNBase): # Simple CNN (8 conv, linear, 128 filters, 3 dropouts, small frame embedding of 128) w/ Soft voting
    def __init__(self, num_classes, input_size, num_channels, num_frames_model, bias, dropout, **kw):
        super().__init__(num_classes, input_size, num_channels, num_frames_model, bias, dropout)        
        cnn_down_factor = 8
        nfil = 128
        self.cnn_out_channel = 2
        self.cnn_out_shape = (int(self.input_size[0]/cnn_down_factor), int(self.input_size[1]/cnn_down_factor))
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, nfil, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(nfil, nfil, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(nfil, nfil, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(nfil, self.cnn_out_channel, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(nfil, self.cnn_out_channel, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.drop1 = nn.Dropout2d(self.p)
        self.drop2 = nn.Dropout2d(self.p)
        self.drop3 = nn.Dropout2d(self.p)

        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + self.drop1(c)
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + self.drop2(c)

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        c = cp + self.drop3(c)
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs