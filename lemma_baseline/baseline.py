#Written by Omer Levy, modified by Javad Hosseini
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
debug = False

from .qa_utils import get_lemmas_only_verbs, get_lemmas_no_stopwords, get_lemmas, get_tuples

class Baseline:

    def __init__(self):
        self.negations = set(['no', 'not', 'never'])

    def run(self, test):
        lemma_intersection = np.array([self.lemma_intersection(q, a) for q, a, v in test])
        matching_voice = np.array([self.matching_voice(q, a) for q, a, v in test])
        same_negation = np.array([self.same_negation(q, a) for q, a, v in test])
        return lemma_intersection * matching_voice * same_negation

    @staticmethod
    def lemma_intersection(q, a):
        q_lemmas_only_verbs = get_lemmas_only_verbs(q[1])
        a_lemmas_only_verbs = get_lemmas_only_verbs(a[1])
        q_lemmas_no_stopwords = get_lemmas_no_stopwords(q[1])
        a_lemmas_no_stopwords = get_lemmas_no_stopwords(a[1])

        share_one_verb = len(q_lemmas_only_verbs.intersection(a_lemmas_only_verbs)) > 0
        answer_contains_all_contents = q_lemmas_no_stopwords == q_lemmas_no_stopwords.intersection(a_lemmas_no_stopwords)
        return share_one_verb and answer_contains_all_contents

    def matching_voice(self, q, a):
        return self.same_voice(q, a) == self.aligned_args(q, a)

    def same_voice(self, q, a):
        q_passive = self.is_passive(q[1])
        a_passive = self.is_passive(a[1])
        return q_passive == a_passive

    @staticmethod
    def is_passive(pred):
        words = get_lemmas(pred)
        be = 'be' in words
        by = 'by' in words

        # Added by Javad
        if len(words)==2 and by:
            return True

        return be and by

    @staticmethod
    def aligned_args(q, a):
        if debug:
            print (q,a)
        q_arg = get_lemmas_no_stopwords(q[2], wn.NOUN)
        if q_arg == get_lemmas_no_stopwords(a[2], wn.NOUN):
            return True
        if q_arg == get_lemmas_no_stopwords(a[0], wn.NOUN):
            return False
        return Baseline.aligned_args(a,q)
        #raise Exception('HORRIBLE BUG!!!', q, " ", a)

    def same_negation(self, q, a):
        q_negated = self.is_negated(q[1])
        a_negated = self.is_negated(a[1])
        return q_negated == a_negated

    def is_negated(self, pred):
        words = get_lemmas(pred)
        return len(set(words).intersection(self.negations)) > 0

class Baseline_sherliic:

    def __init__(self):
        self.sherliic_root = "../../gfiles/ent/sherliic/"
        self.relation_index_path = self.sherliic_root + "relation_index.tsv"
        self.stop_words = stopwords.words('english')
        self.relation_index = self.load_relation_index(self.relation_index_path)

    def run(self, sherliic_path):
        f = open(sherliic_path)
        lines = open(sherliic_path).readlines()[1:]
        predictions = []
        for line in lines:
            ss = line.split(",")
            id_prem = ss[2]
            id_hypo = ss[4]
            is_premise_reversed = ss[13].lower() == "true"
            is_hypothesis_reversed = ss[14].lower() == "true"
            predictions.append(self.lemma(id_prem, id_hypo, is_premise_reversed, is_hypothesis_reversed))
        return predictions



    def lemma(self, id_prem, id_hypo, is_premise_reversed, is_hypothesis_reversed):

        pr_lemmata = self.words_from_id(id_prem)
        hy_lemmata = self.words_from_id(id_hypo)

        print ('prem: ', pr_lemmata)
        print ('hypo: ', hy_lemmata)

        # 1. Criterion: has prem all content words of hypo?
        all_content_words_there = True
        for w in hy_lemmata:
            if w in self.stop_words:
                continue
            if w not in pr_lemmata:
                all_content_words_there = False
                break

        # 2. Criterion: is predicate the same?
        pr_pred = pr_lemmata[-1] if is_premise_reversed else pr_lemmata[0]
        hy_pred = hy_lemmata[-1] if is_hypothesis_reversed else hy_lemmata[0]
        same_predicate = pr_pred == hy_pred

        # 3. Criterion: is voice and inversement the same?
        voice_pr = self.voice_of_id(id_prem)
        voice_hy = self.voice_of_id(id_hypo)
        same_voice = voice_pr == voice_hy
        same_inversement = is_premise_reversed == is_hypothesis_reversed
        third_criterion = same_voice == same_inversement

        return all_content_words_there and same_predicate and third_criterion

    def words_from_id(self, rel_id):
        rel_path = self.get_path(rel_id)
        return [
            w
            for i, w in enumerate(rel_path.split("___"))
            if i % 2 == 1
        ]

    def get_path(self, rel_id):
        return self.relation_index[int(rel_id)]

    def load_relation_index(self, index_file):
        self.relation_index = {}
        with open(index_file) as f:
            for line in f:
                idx, rel = line.strip().split("\t")
                self.relation_index[int(idx)] = rel
        return self.relation_index

    def voice_of_id(self, rel_id):
        path = self.get_path(rel_id)
        if path.startswith("nsubjpass") or path.endswith("nsubjpass") or path.endswith("nsubjpass^-"):
            return "passive"
        else:
            return "active"


def predict_lemma_baseline(fname, args):
    if not fname:
        return None
    if args.dev_sherliic_v2 or args.test_sherliic_v2:
        b = Baseline_sherliic()
        if args.dev_sherliic_v2:
            prediction = b.run(b.sherliic_root + "dev.csv")
        else:
            prediction = b.run(b.sherliic_root + "test.csv")
    else:
        test = get_tuples(fname)
        b = Baseline()
        prediction = b.run(test)

    return prediction
