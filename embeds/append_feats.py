import sys

#
# append_feats.py <feats-file> <new-feat-file>
#
# Assumes the first given file is Javad's full file of features (feats_rel.txt).
# The second file should have the same number of lines as the first, but with just
# a single number on each line. That is the new feature to append to the end of
# Javad's feats file.
#
# OUTPUT: a new file called 'merged-feats.txt'
#


outfile = "merged-feats.txt"

allfeatsfile = sys.argv[1]
with open(allfeatsfile) as f:
    allfeatslines = f.readlines()

newfeatsfile = sys.argv[2]    
with open(newfeatsfile) as f:
    newfeatslines = f.readlines()


print "Writing new file:", outfile
ii = 0
of = open(outfile, 'w')
for line in allfeatslines:
    # (send.2,send.to.2)#(replace.1,replace.2)#False	7598	[  1.59694555e-01   1.35729080e-01   4.59848780e-02   3.60377257e+01  1.56236541e+01   5.28306838e-02   4.37096582e-02   2.26846643e-02  2.45422812e-02   6.85719704e-02   7.23620376e-02   1.05148568e-02  4.17585008e-02   5.45159318e-03   1.71009240e-02   4.07884856e-03  4.95666005e-03   4.44824242e-03   4.19624643e-03   5.50417944e-03  5.99910627e-03   6.92461394e-03]

    line = line.rstrip()
    linetrim = line[0:-1]
    of.write(linetrim)
    of.write("   ")
    of.write(newfeatslines[ii].rstrip())
    of.write("]\n")
    ii += 1
of.close()
