###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

# Volkan Okbay (2022)
# Creates blacklist[threshold].yaml files under blacklists folder. Takes results of slowfast and x3dm model from
# corresponding .csv files.

import numpy as np
import csv
import yaml

blacklist_thr = 50 # Only parameter. Blacklist samples with a score under this value. 

scores = np.zeros((5728 ,3))

# Slowfast scores (evaluate 32 frames/video, score: 50)
with open('./datacentric results/slowfast_train.csv') as f:
    reader = csv.reader(f)
    for n, row in enumerate(reader):
        vidx = row[0].split(';')[0]
        lab = row[0].split(';')[1]
        pred = row[0].split(';')[2]
        scores[n,0] = vidx
        scores[n,1] = lab
        scores[n,2] += (lab == pred)*50

# X3Dm scores (evaluate 3x16 frames/video, score: 12.5 + 25 + 12.5)
with open('./datacentric results/x3dm_train.csv') as f:
    reader = csv.reader(f)
    count_to_3 = 0
    for n, row in enumerate(reader):
        lab = row[0].split(';')[1]
        pred = row[0].split(';')[2]
        if count_to_3 == 0:
            scores[int(n/3),2] += (lab == pred)*12.5
            count_to_3 += 1
        elif count_to_3 == 1:
            scores[int(n/3),2] += (lab == pred)*25
            count_to_3 += 1
        elif count_to_3 == 2:
            scores[int(n/3),2] += (lab == pred)*12.5
            count_to_3 = 0

# Thresholding out 100 points
black_pul = []
black_psh = []
black_sit = []
black_sqt = []
black_neg = []
for row in scores:
    if row[2] < blacklist_thr:
        if row[1] == 0: black_pul.append(int(row[0]))
        elif row[1] == 1: black_psh.append(int(row[0]))
        elif row[1] == 2: black_sit.append(int(row[0]))
        elif row[1] == 3: black_sqt.append(int(row[0]))
        elif row[1] == 4: black_neg.append(int(row[0]))

# Create dictionary and write yaml file       
dict1 = {'pull ups': black_pul, 'push up': black_psh, 'situp': black_sit, 'squat': black_sqt, 'background': black_neg}
with open(f'../blacklists/blacklist{blacklist_thr}.yaml', 'w') as f:
    yaml.dump(dict1, f)