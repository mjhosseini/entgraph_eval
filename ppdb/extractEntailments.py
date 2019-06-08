import sys
sys.path.append("..")
from evaluation.util import read_data_plain


onlyEntailment = True

def write_predictedLabel(out_path, lines_no_label, pqs, entpqs):
    f = open(out_path,'w')
    for idx in range(len(pqs)):
        if pqs[idx] in entpqs:
            f.write(lines_no_label[idx]+'\t'+'True\n')
        else:
            f.write(lines_no_label[idx] + '\t' + 'False\n')

    f.close()

#return the subset of pqset that are entailments in ppdb2
def getEntailments(pqset, ppdbPath):
    f = open(ppdbPath)
    idx = 0
    ret = set()
    for line in f:

        idx+=1
        ss = line.split("|||")

        if onlyEntailment:
            label = ss[-1].strip().lower()
            if label!="equivalence" and label!='ForwardEntailment':
                continue

        pq = ss[1].strip()+"#"+ss[2].strip()

        if idx % 100000 == 0:
            print idx

        # if idx>1000000:
        #     break
        if pq in pqset:
            print "found some: ", pq
            ret.add(pq)
    return ret

root = "../../gfiles/"
orig_fname = root+'ent/all_new_comb.txt'
out_fname = root+'ent/all_new_comb_ppdb_xl_ent.txt'
lines_no_label, pqlist, _ = read_data_plain(orig_fname)
pqset = set(pqlist)
print pqset

s1 = getEntailments(pqset,root+'ppdb-2.0-xl-phrasal')
s2 = getEntailments(pqset,root+'ppdb-2.0-xl-lexical')
entpqset = s1.union(s2)

write_predictedLabel(out_fname,lines_no_label,pqlist,entpqset)