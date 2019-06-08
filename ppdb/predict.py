from evaluation.util import read_data_plain

def predict(resourcePath,fname):

    _,pqs_resource,l_resource = read_data_plain(resourcePath)
    _,pqs_test,_ = read_data_plain(fname)

    resource_true = set()

    for idx,pq in enumerate(pqs_resource):
        if l_resource[idx]:
            resource_true.add(pq)
    ret = []
    for pq in pqs_test:
        if pq in resource_true:
            print "has pq: ", pq
            ret.append(True)
        else:
            ret.append(False)

    return ret
