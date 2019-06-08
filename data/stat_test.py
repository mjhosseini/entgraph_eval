import sys
sys.path.append("..")

from evaluation import util

args = sys.argv[1:]
root = "../../gfiles/ent/"
Y, Y_pred1 = util.read_Ys(root+args[0])
Y,Y_pred2 = util.read_Ys(root+args[1])
print Y
util.statistical_significance(Y,Y_pred1,Y_pred2)
