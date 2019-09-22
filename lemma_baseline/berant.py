# Written by Omer Levy, modified by Javad Hosseini
from collections import defaultdict

from .qa_utils import aligned_args, get_lemmas, get_tuples

debug = False


class BerantRules:
    # resource2 = True: Use 2012 resource (~15m rules) instead of 2011 (~30K rules)
    def __init__(self, path, context_sensitive=False, resource2=False, berDS=False):
        if not resource2:
            self.db = self.load_db(path + 'berant_rules.txt')
        else:
            self.db = self.load_db2(path + 'reverb_global_clsf_all_tncf_lambda_0.1.txt')
        self.context_sensitive = context_sensitive
        self.berDS = berDS
        if context_sensitive:
            self.types = self.load_types(path + 'allclassinstances.txt')

    def load_db(self, db_path):
        db = defaultdict(list)
        with open(db_path) as fin:
            for line in fin:
                lhs, rel, rhs = line.strip().split('\t')
                lhs, lhs_slots = self.parse(lhs)
                rhs, rhs_slots = self.parse(rhs)

                if (lhs is None) or (rhs is None) or len(lhs_slots) != len(rhs_slots):
                    raise Exception('HORRIBLE BUG!!!')

                if rel == '->':
                    db[(lhs, rhs)].append((lhs_slots == rhs_slots, lhs_slots))
                elif rel == '-R>':
                    db[(lhs, rhs)].append((False, lhs_slots))
                else:
                    raise Exception('HORRIBLE BUG!!!')
        return db

    def load_db2(self, db_path):
        if debug:
            print ("loading db2")
        db = defaultdict(list)
        with open(db_path) as fin:
            for line in fin:
                # print "b line: ", line
                aligned = (line.count("@R@") != 1)
                line = line.replace("@R@", "")
                # print "now line: ", line
                lhs, rhs = line.strip().split('\t')

                # print "l,r: ",lhs, rhs
                if debug:
                    print (aligned)
                db[(lhs, rhs)].append((aligned, ('thing', 'thing')))

        return db

    @staticmethod
    def parse(pattern):
        pred, x, y = pattern[1:-1].split('::')
        return pred, (x, y)

    @staticmethod
    def load_types(types_path):
        with open(types_path) as fin:
            lines = [line.strip().split('\t') for line in fin]
        types = defaultdict(list)
        for line in lines:
            if len(line) >= 2:
                types[line[1]].append(line[0])
        return types

    def prob_entailing(self, q, a):
        aligned = aligned_args(q, a)
        if debug:
            print ("berant aligned: ", aligned)
        if aligned == -1:
            aligned = aligned_args(a, q)
            if aligned == -1:
                raise Exception('HORRIBLE BUG!!!' + str(q) + " " + str(a))
        q_pred, a_pred = (' '.join(get_lemmas(q[1])), ' '.join(get_lemmas(a[1])))
        if ((a_pred, q_pred) in self.db):
            slots = set([lhs_slots for alignment, lhs_slots in self.db[(a_pred, q_pred)] if aligned == alignment])
            if len(slots) > 0:
                if not self.context_sensitive:
                    return 1.0
                else:
                    if not self.berDS:
                        possible_slots = set([(x, y) for x in self.types[a[0]] for y in self.types[a[2]]])

                    else:
                        t1 = a[0].replace("_1", "").replace("_2", "")
                        t2 = a[2].replace("_1", "").replace("_2", "")
                        if debug:
                            print ("t1, t2: ", t1, t2)
                        possible_slots = set([(t1, t2)])
                    if len(slots.intersection(possible_slots)) > 0:
                        return 1.0
        return 0.0


def predict_Berant(path, fname, context=False, resource2=False, berDS=False):
    test = get_tuples(fname)
    b = BerantRules(path, resource2=resource2, context_sensitive=context, berDS=berDS)
    prediction = [b.prob_entailing(q, a) for (q, a, _) in test]

    return prediction