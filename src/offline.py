###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# This script is intended for remote-ssh training run in background.

import os
import config

cfg = config.ConfigParser()
trainNo = cfg.read("trainNo")
os.system("nohup python3 train.py >./log/resNo" + str(trainNo) + ".txt 2>&1 &")
