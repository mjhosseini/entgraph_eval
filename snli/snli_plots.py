import matplotlib.pyplot as plt
import numpy as np
import os

def get_color(method_name,idx):
    print "method_name:", method_name
    method_name = method_name.lower()
    hatch = "-"
    idxUsed = False

    if (method_name == "binc_cg_pr" or method_name == "weedpr_cg_pr"):
        color = 'r'
        marker = '^'

    elif (method_name == "binc_cg" or method_name == "weedpr_cg"):
        color = 'purple'
        marker = '1'

    elif (method_name == "binc" or method_name=="weedpr"):
        color = 'pink'
        marker = 'v'
        hatch = "*"
    elif (method_name == "binc_avg" or method_name == "weedpr_avg"):
        color = 'orange'
        marker = 's'
        hatch = '\\'
    elif (method_name == "binc_untyped" or method_name == "weedpr_untyped"):
        color = 'g'
        marker = 'x'
    elif (method_name == "lemma_baseline"):
        color = 'k'
        marker = 'o'
    elif (method_name == "berant_ilp"):
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
            color = 'green'
            marker = '^'
            hatch = '\\'

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
            color = 'brown'
            marker = 'o'
        else:
            color = 'k--'
            marker = 'x'
    print color
    return (color, marker, hatch,idxUsed)

def get_color_by_idx(idx):
    print "idx:", idx
    # method_name = method_name.lower()
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
    else:
        color = 'k--'
        marker = 'x'

    return (color, marker, hatch)

def draw_plot(fname):
    print fname


    f = open(root + fname)

    header = f.readline().strip("\n")
    methods = header.split(",")[1:]
    tmp = methods[1]
    methods[1] = methods[2]#TODO: remove this
    methods[2] = tmp

    print methods

    accs_list = [[] for _ in range(len(methods)+1)]

    print accs_list

    for line in f:
        print line.strip("\n").split(",")
        this_accs = [np.float(x) for x in line.split(",")]
        if sum(this_accs[1:])==0:
            continue
        [accs_list[i].append(x) for i,x in enumerate(this_accs)]

    for i, method in enumerate(methods):
        color, marker, hatch, idxUsed = get_color(method, i)
        label = method
        ms = 5
        plt.plot(accs_list[0], accs_list[i+1], color, label=label, marker=marker, linewidth=1, ms=ms)

    plt.xlabel("Bigram Overlap Ratio", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)

    plt.legend(loc=1, prop={'size': 8})
    plt.savefig(root + "plots/"+fname[:-4]+".png")

root = "../../gfiles/snli_1.0/overlap_accs/"


fnames = os.listdir(root)
for fname in fnames:
    if not fname.endswith(".txt"):
        continue
    plt.clf()
    plt.cla()
    draw_plot(fname)