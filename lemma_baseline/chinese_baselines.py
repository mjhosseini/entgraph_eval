# coding=utf-8
# Written by Omer Levy, modified by Javad Hosseini, Tianyi Li
import numpy as np
# from nltk.corpus import wordnet as wn
# from nltk.corpus import stopwords

from .qa_utils_chinese import get_tuples, get_raws
# from lemma_baseline.StanfordCoreNLP import *


'''
class Chinese_Baseline:
    def __init__(self):
        self.negations = {'不', '从未', '未能'}
        StanfordCoreNLP_chinese_properties = get_StanfordCoreNLP_chinese_properties()

        self.corenlpClient = CoreNLPClient(
                annotators=['tokenize', 'ssplit', 'pos', 'lemma'],
                properties=StanfordCoreNLP_chinese_properties,
                timeout=30000,
                memory='16G')

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
'''

class Chinese_Naive_Baseline:
    def __init__(self):
        pass

    def run(self, test):
        return np.array([q[0] == a[0] and q[1] == a[1] and q[2] == a[2] for q,a,v in test])

    def run_predonly(self, test):
        return np.array([q[1] == a[1] for q,a,v in test])

    def run_exact(self, test):
        return np.array([prem == hypo for prem,hypo in test])


def predict_lemma_baseline(fname, args):
    if not fname:
        return None
    test = get_tuples(fname)
    b = Chinese_Naive_Baseline()
    prediction = b.run(test)
    return prediction


def predict_coarse_lemma_baseline(fname, args):
    if not fname:
        return None
    test = get_tuples(fname)
    b = Chinese_Naive_Baseline()
    prediction = b.run_predonly(test)
    return prediction


def predict_exact_baseline(fname, args):
    if not fname:
        return None
    test = get_raws(fname)
    b = b = Chinese_Naive_Baseline()
    prediction = b.run_exact(test)
    return prediction

