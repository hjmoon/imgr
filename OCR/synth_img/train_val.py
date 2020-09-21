import numpy as np
import os

np.random.seed(2)
data = {}
train = []
val = []
with open('results/gt.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        path, labels, rois = line.split('\t')
        if path in data:
            data[path]['labels'].append(labels)
            data[path]['rois'].append(rois)
        else:
            data[path]={'labels':[labels],
                        'rois':[rois]}

data_ind = [v for v in range(len(data))]

train_size = int(len(data_ind) * 0.7)
val_size = len(data_ind)-train_size

train_f = open('results/train/gt.txt', 'w', encoding='utf-8')
val_f = open('results/val/gt.txt', 'w', encoding='utf-8')

for data_ind, path in enumerate(data):
    src = path
    if data_ind < train_size:
        split = 'train'
    else:
        split = 'val'
    filename = path.split('/')[-1]
    dst = os.path.join(split,'images',filename)
    for label_ind in range(len(data[path]['labels'])):
        labels = data[path]['labels'][label_ind]
        rois = data[path]['rois'][label_ind]
        if data_ind < train_size:
            train_f.write(dst+'\t'+labels+'\t'+rois+'\n')
        else:
            val_f.write(dst + '\t' + labels + '\t' + rois + '\n')
train_f.close()
val_f.close()