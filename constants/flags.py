def opts(actual_args=None):
    '''
    Parse program arguments. If actual_args is None, argparse will refer to sys.argv.
    Otherwise, actual_args should be a list of strings.
    '''

    import argparse

    opts = argparse.ArgumentParser(description='Learn or predict from a discriminative tagging model')

    def flag(name, description, ftype=str, **kwargs):
        opts.add_argument(('--' if len(name)>1 else '-')+name, type=ftype, help=description, **kwargs)
    def inflag(name, description, ftype=argparse.FileType('r'), **kwargs):
        flag(name, description, ftype=ftype, **kwargs)
    def outflag(name, description, ftype=argparse.FileType('w'), **kwargs):
        flag(name, description, ftype=ftype, **kwargs)
    def boolflag(name, description, default=False, **kwargs):
        opts.add_argument(('--' if len(name)>1 else '-')+name, action='store_false' if default else 'store_true', help=description, **kwargs)

    boolflag("write", "whether we should write pr_rec, auc and tp_fp")
    boolflag("no_lemma_baseline", "should we ignore lemma_baseline?")
    boolflag("no_constraints", "should we ignore constraints?")
    boolflag("saveMemory", "If yes, we'll only consider the BINC feature not all, in the graph")
    flag("gpath", "The path to entailment graph folders")
    flag("method", "method's name (used for writing output)")
    flag("CCG", "Whether it's CCG parsing or openIE",ftype=int)
    flag("typed", "Whether it's typed",ftype=int)
    flag("supervised", "Whether it's supervised",ftype=int)
    flag("oneFeat", "Whether it's only BINC, or avg of features",ftype=int)
    # flag("Gbooks", "Whether it's Gbooks or NewsSpike. Only works for openIE",ftype=int)
    flag("useSims", "whether we should use similar rels",ftype=int)
    flag("featsFile", "name of features file",ftype=str)
    flag("featIdx", "if anything other than 4!",ftype=int)
    flag("wAvgFeats", "weighted average of feats", ftype=str)#e.g., 0 .4 4 .6, means features 0 and 4 with the weights
    flag("gAvgFeats", "geometric average of feats", ftype=str)#e.g., 0 4, means features 0 and 4
    boolflag("backupAvg", "backup to average for out of graph predicates")
    boolflag("dev", "only on dev data")
    boolflag("test", "on test data")
    boolflag("dev_v2", "only on dev data, version 2")#version 2 built on 04/04/2019
    boolflag("test_v2", "on test data, version 2")#version 2 built on 04/04/2019
    boolflag("dev_v3", "only on dev data, version 2")  # version 2 built on 04/04/2019
    boolflag("test_v3", "on test data, version 2")  # version 2 built on 04/04/2019
    boolflag("dev_dir", "on dev_dir data")
    boolflag("test_dir", "on test_dir data")
    boolflag("test_naacl", "on naacl's data")
    boolflag("test_naacl_untensed", "on naacl's data, untensed parse")
    boolflag("zeichner", "on zeichner's data")
    boolflag("dev_sherliic_v2", "on sherliic's data dev portion, v2 rels")
    boolflag("test_sherliic_v2", "on sherliic's data test portion, v2 rels")
    boolflag("snli", "extracts feats for snli ds")
    boolflag("instance_level", "vs typed pred-pair level that we usually do.")#Useful for glove embedding or other embedding predictions

    boolflag("tnf", "whether we should learn tnf")
    boolflag("exactType", "Whether we should use the exact same type as query or average between different types")#Use the exact featres for inference (the first half that could be potentially from typeProp)?# Not for TNF
    boolflag("rankFeats", "Whether we should use the feature or 1/rank")#,ftype=float
    boolflag("rankDiscount", "Whether we should multiply the feature by 1/sqrt(rank)")
    flag("maxRank","maximum number of neighbors to read for a node", ftype=int)
    flag("threshold", "threshold small edges",ftype=float)

    boolflag("berDS", "predict on berant's DS")
    boolflag("berDS_v2", "predict on berant's DS")
    boolflag("berDS_v3", "predict on berant's DS")
    boolflag("origLevy", "predict on original levy set")
    boolflag("calcSupScores", "calculate all supervised graph sores")
    flag("dsPath", "optionally, provide a test set path")
    flag("outDir", "optionally, where to output the pr_recs")
    flag("sim_suffix", "similarity files suffix", ftype=str)
    boolflag("debug", "writing debug info")

    boolflag("LDA", "full distributional LDA probabilities for types")  # deprecated

    args = opts.parse_args(actual_args)
    return args
