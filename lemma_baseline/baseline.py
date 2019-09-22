#Written by Omer Levy, modified by Javad Hosseini
import numpy as np
from nltk.corpus import wordnet as wn
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

def predict_lemma_baseline(fname):
    if not fname:
        return None
    test = get_tuples(fname)
    b = Baseline()
    prediction = b.run(test)

    return prediction
