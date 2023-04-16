import torch as tr
import numpy

def compute_acc(outputs, labels):
    pred = numpy.array(outputs)
    labels_np = numpy.array(labels)
    total = len(labels)
    correct = sum(pred == labels_np)
    return total, correct

def compute_weighted_acc(outputs, labels, cls_w):
    pred = numpy.array(outputs)
    labels_np = numpy.array(labels)
    cls_w_np = numpy.array(cls_w)
    w_ind = labels_np - 1
    wl = cls_w_np[w_ind]
    total = sum(wl)
    correct = sum((pred == labels_np)*wl)
    return total, correct

o = [3,1,2,1,1]
l = [tr.tensor(3), tr.tensor(2), tr.tensor(2), tr.tensor(1), tr.tensor(3)]
w = [0.1 ,0.5, 0.4]

a,b = compute_acc(o,l)
print(b/a*100)

a,b = compute_weighted_acc(o,l,w)
print(b/a*100)