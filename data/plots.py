import matplotlib.pyplot as plt
import sys
from sklearn import metrics
import numpy as np
import os

args = sys.argv[1:]
aucs = []

def auc_exact_below(xs,ys):
    ret = 0
    sortIdx = np.argsort(xs)
    print sortIdx
    xs = [xs[i] for i in sortIdx]
    ys = [ys[i] for i in sortIdx]

    for i in range(1,len(xs)):
        dx = xs[i]-xs[i-1]
        y = (ys[i]+ys[i-1])/2
        ret += (dx*y)
    return ret

def get_color(method_name,idx):
    print "method_name:", method_name
    method_name = method_name.lower()
    hatch = "-"
    idxUsed = False

    if (method_name == "binc_cg_pr" or method_name == "weedpr_cg_pr" or method_name == "global binc" or method_name == "local binc"):
        color = 'r'
        marker = '^'

    elif (method_name == "binc_cg" or method_name == "weedpr_cg" or method_name == "conve_rw"):
        color = 'purple'
        marker = '1'

    elif (method_name == "binc" or method_name=="weedpr" or method_name=="local binc"):
        color = 'pink'
        marker = 'v'
        hatch = "*"
    elif (method_name == "binc_avg" or method_name == "weedpr_avg" or  method_name == "conve_rw_cos" or method_name.endswith("aug mc")):
        color = 'orange'
        marker = 's'
        hatch = '\\'
    elif (method_name == "binc_untyped" or method_name == "weedpr_untyped"):
        color = 'g'
        marker = 'x'
    elif (method_name == "lemma_baseline"):
        color = 'k'
        marker = 'o'
    elif (method_name == "berant_ilp" or method_name == "berant ilp"):
        color = 'black'
        marker = 'D'
    elif (method_name == "ppdb"):
        color = 'brown'
        marker = 'D'
    elif (method_name == "binc_uniformm"):
        color = 'brown'
        marker = 'x'
    else:
        idxUsed = True
        if (idx==0):
            color = 'b'
            marker = 'x'


        elif (idx == 1):
            color = 'brown'
            marker = 'o'


        elif (idx==2):
            color = 'lightblue'
            marker = '1'
            # hatch = "-"

        elif (idx==3):
            color = 'purple'
            marker = '1'
        elif (idx==4):
            color = 'b'
            marker = 'o'
        elif (idx==5):
            color = 'lightblue'
            marker = 'v'

        elif (idx==6):
            color = 'green'
            marker = '^'
            hatch = '\\'
        else:
            color = 'k--'
            marker = 'x'
    print color
    return (color, marker, hatch,idxUsed)

def get_color_by_idx(idx):
    print "idx:", idx
    hatch = "-"

    if (idx==0):
        color = 'c'
        marker = '^'
        hatch = "/"
    elif (idx==1):
        color = 'r'
        marker = 's'
        hatch = "*"
    elif (idx==2):
        color = 'r'
        marker = 'v'
    elif (idx==3):
        color = 'g'
        marker = 'o'
        hatch = "||"
    elif (idx==3):
        color = 'k'
        marker = 'o'
    elif (idx==4):
        color = 'b'
        marker = 'x'
        hatch = '\\'
    elif (idx==5):
        color = 'b'
        marker = 'v'

    elif (idx==6):
        color = 'lightblue'
        marker = '1'
        hatch = "-"
    elif (idx==7):
        color = 'lightblue'
        marker = 'v'
    elif (idx==8):
        color = 'purple'
        marker = 'D'
    elif (idx==9):
        color = 'b'
        marker = 'o'
    elif (idx==10):
        color = 'brown'
        marker = 'D'
    elif (idx==11):
        color = 'brown'
        marker = 'o'
# #        hatch = "D"
#     elif (method_name == "grabr wo overlap"):
#         color = 'y'
#         marker = 'o'

    else:
        color = 'k--'
        marker = 'x'

    return (color, marker, hatch)



root = "../../gfiles/results/dev/boxemb/"


if args:
    root = "../../gfiles/results/"+ args[0]+"/"

pr_rec = not "tp_fp" in root

plt.clf()
plt.cla()

fnames = os.listdir(root)
fnames.reverse()
f_idx = 0
usedIdx = 0
for fname in fnames:

    if not fname.endswith(".txt"):
        continue
    print fname

    method = fname[:-4]
    f = open(root+fname)
    xs = []
    ys = []
    x2s = []
    y2s = []
    lines = f.read().splitlines()

    best_f1 = 0
    best_pr = 0
    best_rec = 0

    for (i,line) in enumerate(lines):
        print ("line:", line)
        if len(line.split())<2:
            continue
        if line.startswith("auc"):
            auc = line.split()[1]
            continue

        mod = 20

        ss = line.split()
        pr = float(ss[0])
        rec = float(ss[1])

        try:
            f1 = 2 * pr * rec / (pr + rec)
        except:
            f1 = 0


        if f1 > best_f1:
            best_f1 = f1
            best_pr = pr
            best_rec = rec

        if pr > .5 and i != len(lines) - 1:
            x2s.append(rec)
            y2s.append(pr)


        if len(lines)<30 or i%mod==0 or i==1 or (pr>.7 and i%(mod/2)==0) or (pr>.8 and i%(mod/3)==0) or (pr>.8 and i%(mod/4)==0) or i>len(lines)-3:
            xs.append(rec)
            ys.append(pr)

    auc = -1
    if (len(x2s)>2):
        auc = metrics.auc(x2s, y2s, reorder=True)
        print "pr_rec, auc: ", auc, method
        aucs.append(auc)

        my_auc = auc_exact_below(x2s,y2s)
        print "my_auc:", my_auc
    elif "berant" in method.lower():
        aucs.append(-1)
    elif "ppdb" in method.lower():
        aucs.append(-2)
    else:
        aucs.append(-3)

    print xs
    print ys
    xs = xs[0:len(xs)-1]
    ys = ys[0:len(ys) - 1]
    (color,marker,hatch,idxUsed) = get_color(method,usedIdx)
    if idxUsed:
        usedIdx += 1

    label = method
    auc = str(auc)
    if auc!="-1":
        print "auc is: ", auc, method
        label += ": "+str(auc[0:7])
    ms = 5
    if len(xs)==1:
        ms = 8
    if len(xs)>1:
        plt.plot(xs, ys, color, label =label, marker=marker, linewidth=1, ms = ms)
    else:
        plt.plot(xs[0], ys[0], color, label=label, marker=marker, ms=ms,linewidth=1,linestyle="None")

    print "best f1, pr, rec: "
    print best_f1, best_pr, best_rec

    f_idx += 1


if 'ber' in root:

    plt.xlim([.08, .5])  # changee
    plt.ylim([.3, 1])  # changee
else:
    # plt.xlim([.13, .37])#changee
    # plt.ylim([.5, .85])#changee

    # plt.xlim([.1, .5])#changee
    # plt.xlim([.13, .45])  # changee
    # plt.xlim([.13, .63])  # changee
    plt.ylim([.3, .85])#changee
    # plt.xlim([.11, .55])  # changee
    pass

if pr_rec:
    plt.xlabel("Recall", fontsize = 20)
    plt.ylabel("Precision", fontsize = 20)#Change
    # plt.legend(loc = 1, prop={'size': 20})#was 14

    handles, labels = plt.gca().get_legend_handles_labels()
    # order = [2, 3,4, 1,5,6,0]
    order = np.argsort(aucs)
    order = order[::-1]

    # handles[0] = mpatches.Patch(color='red')
    # order = [1, 3, 0, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc = 1, prop={'size':6})#was 11
    if args:
        plt.savefig(root + args[1] + "_pr_rec.png")
    else:
        plt.savefig(root + "pr_rec.png")

else:
    plt.xlabel("FP", fontsize = 20)
    plt.ylabel("TP", fontsize = 20)#Change
    plt.legend(loc = 4, prop={'size':9})
    plt.savefig(root+"tp_fp.png")
