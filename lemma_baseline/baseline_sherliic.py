#mainly from https://github.com/mnschmit/SherLIiC/

from nltk.corpus import stopwords

def lemma(id_prem, id_hypo, is_premise_reversed, is_hypothesis_reversed):
    stop_words = stopwords.words('english')
    pr_lemmata = words_from_id(id_prem)
    hy_lemmata = words_from_id(id_hypo)

    # 1. Criterion: has prem all content words of hypo?
    all_content_words_there = True
    for w in hy_lemmata:
        if w in stop_words:
            continue
        if w not in pr_lemmata:
            all_content_words_there = False
            break

    # 2. Criterion: is predicate the same?
    pr_pred = pr_lemmata[-1] if is_premise_reversed else pr_lemmata[0]
    hy_pred = hy_lemmata[-1] if is_hypothesis_reversed else hy_lemmata[0]
    same_predicate = pr_pred == hy_pred

    # 3. Criterion: is voice and inversement the same?
    voice_pr = voice_of_id(id_prem)
    voice_hy = voice_of_id(id_hypo)
    same_voice = voice_pr == voice_hy
    same_inversement = is_premise_reversed == is_hypothesis_reversed
    third_criterion = same_voice == same_inversement

    return all_content_words_there and same_predicate and third_criterion


def words_from_id(rel_id, relation_index_path="../../gfiles/ent/sherliic/relation_index.tsv"):
    rel_path = get_path(rel_id, relation_index_path=relation_index_path)
    return [
        w
        for i, w in enumerate(rel_path.split("___"))
        if i % 2 == 1
    ]

def get_path(rel_id, relation_index_path="data/relation_index.tsv"):
    global relation_index
    if relation_index is None:
        relation_index = load_relation_index(relation_index_path)

    return relation_index[int(rel_id)]

def load_relation_index(index_file):
    relation_index = {}
    with open(index_file) as f:
        for line in f:
            idx, rel = line.strip().split("\t")
            relation_index[int(idx)] = rel
    return relation_index

def voice_of_id(rel_id):
    path = get_path(rel_id)
    if path.startswith("nsubjpass") or path.endswith("nsubjpass") or path.endswith("nsubjpass^-"):
        return "passive"
    else:
        return "active"

relation_index = None
lemma_pred = lemma(182478, 65184, False, False)
print (lemma_pred)
