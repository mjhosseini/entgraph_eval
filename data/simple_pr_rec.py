import numpy as np
from sklearn import metrics

Y_dev = []
Y_dev_pred = []

root = "../../gfiles/ent/"

f_orig = "naacl_levy_format.txt"
f_pred = "TEA_cos_subset.txt"

for l in open(root + f_orig):
    l = l.strip()
    # print (l)
    ss = l.split("\t")
    Y_dev.append(np.float(ss[2].strip()))

for l in open(root + f_pred):
    Y_dev_pred.append(np.float(l.strip()))

(precision, recall, thresholds) = metrics.precision_recall_curve(Y_dev, Y_dev_pred)

for i,p in enumerate(precision):
    print p, recall[i]
