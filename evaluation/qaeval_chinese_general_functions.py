import json
import psutil
import os
import sys
sys.path.append('/disk/scratch_big/tli/multilingual-lexical-inference/lm-lexical-inference/src/data/')
sys.path.append('/disk/scratch/tli/multilingual-lexical-inference/lm-lexical-inference/src/data/')
sys.path.append('/home/s2063487/multilingual-lexical-inference/lm-lexical-inference/src/data/')
sys.path.append('/Users/teddy/PycharmProjects/multilingual-lexical-inference/lm-lexical-inference/src/data')

sys.path.append("/Users/teddy/eclipse-workspace/EntGraph_nick/evaluate")
sys.path.append('/disk/scratch_big/tli/EntGraph_nick/evaluate')
sys.path.append('/disk/scratch/tli/EntGraph_nick/evaluate')
from graph_encoder import pred_deoverlap
import copy
import random
import transformers
random.seed()
import numpy as np
import torch
import string

from common import LABEL_KEY, SENT_KEY, ANTI_KEY  # this is from discrete

from qaeval_utils import DateManager, parse_rel, calc_simscore, duration_format_print, upred2bow, split_str_multipat, \
                         rel2concise_str, fetch_wn_smoothings
from pytorch_lightning.metrics.functional import auc
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def load_data_entries(fpath, posi_only=False, negi_only=False):
    assert not (posi_only and negi_only)
    data_entries = []
    with open(fpath, 'r', encoding='utf8') as fp:
        for line in fp:
            item = json.loads(line)
            assert item['label'] is not None and isinstance(item['label'], bool)
            # if the posi_only flag is set to True, then don't load the negatives! (This is reserved for wh-question answering (objects))
            if item['label'] is not True and posi_only:
                continue
            if item['label'] is not False and negi_only:
                continue
            data_entries.append(item)
    process = psutil.Process(os.getpid())
    print(f"Current memory usage in bytes: {process.memory_info().rss}")  # in bytes
    return data_entries


def calc_bsln_prec(data_entries):
    posi_cnt = 0.0
    total_cnt = 0.0
    for ent in data_entries:
        if ent['label'] is True:
            posi_cnt += 1
        elif ent['label'] is False:
            pass
        else:
            raise AssertionError
        total_cnt += 1
    bsln_prec = posi_cnt / total_cnt
    print(f"baseline precision: {bsln_prec}!")
    return bsln_prec


def compute_ss_auc(precisions: torch.FloatTensor, recalls: torch.FloatTensor,
                filter_threshold: float = 0.5) -> torch.FloatTensor:
    xs, ys = [], []
    for p, r in zip(precisions, recalls):
        if p >= filter_threshold:
            xs.append(r)
            ys.append(p)

    return auc(
        torch.cat([x.unsqueeze(0) for x in xs], 0),
        torch.cat([y.unsqueeze(0) for y in ys], 0)
    )


def type_matched(types_lst_pointer, tsubj, tobj):
    types_lst = copy.deepcopy(types_lst_pointer)
    assert len(types_lst) == 2
    if types_lst[0][-2:] == '_1':
        assert types_lst[1][-2:] == '_2'
        types_lst[0] = types_lst[0][:-2]
        types_lst[1] = types_lst[1][:-2]
    if types_lst[0][-2:] == '_2':
        assert types_lst[1][-2:] == '_1'
        types_lst[0] = types_lst[0][:-2]
        types_lst[1] = types_lst[1][:-2]

    if tsubj[-2:] == '_1':
        assert tobj[-2:] == '_2'
        tsubj = tsubj[:-2]
        tobj = tobj[:-2]
    if tsubj[-2:] == '_2':
        assert tobj[-2:] == '_1'
        tsubj = tsubj[:-2]
        tobj = tobj[:-2]

    if tsubj == types_lst[0] and tobj == types_lst[1]:
        return True
    elif tsubj == types_lst[1] and tobj == types_lst[0]:
        return True
    else:
        return False


def type_contains(types_lst_pointer, t_indexarg):
    types_lst = copy.deepcopy(types_lst_pointer)
    assert len(types_lst) == 2
    if types_lst[0][-2:] == '_1':
        assert types_lst[1][-2:] == '_2'
        types_lst[0] = types_lst[0][:-2]
        types_lst[1] = types_lst[1][:-2]
    if types_lst[0][-2:] == '_2':
        assert types_lst[1][-2:] == '_1'
        types_lst[0] = types_lst[0][:-2]
        types_lst[1] = types_lst[1][:-2]

    if t_indexarg[-2:] == '_1' or t_indexarg[-2:] == '_2':
        t_indexarg = t_indexarg[:-2]

    if t_indexarg in types_lst:
        return True
    else:
        return False


def mask_entities_for_entries(args):
    data_entries = load_data_entries(args.fpath, posi_only=True)
    out_fp = open(args.wh_fpath, 'w', encoding='utf8')

    for ent in data_entries:

        upred, subj, obj, tsubj, tobj = parse_rel(ent)
        arg2mask = None
        if args.mask_only_objs:
            arg2mask = 'obj'
        else:
            if random.random() > 0.5:
                arg2mask = 'subj'
            else:
                arg2mask = 'obj'

        if arg2mask == 'subj':
            ent['answer'] = subj
            ent['answer_type'] = tsubj
            ent['index_arg'] = obj
            ent['index_position'] = 'obj'
            ent['index_type'] = tobj
        elif arg2mask == 'obj':
            ent['answer'] = obj
            ent['answer_type'] = tobj
            ent['index_arg'] = subj
            ent['index_position'] = 'subj'
            ent['index_type'] = tsubj
        else:
            raise AssertionError

        out_line = json.dumps(ent, ensure_ascii=False)
        out_fp.write(out_line+'\n')

    out_fp.close()


def mask_negi_entities_for_entries(args):
    data_entries = load_data_entries(args.fpath, negi_only=True)

    posi_data_entries = load_data_entries(args.wh_fpath, posi_only=True)

    # There may be entries with the same in_partition_sidx but in different partitions mixed together, but this is okay
    # as long as we check for partition key when looking for the corresponding positives for each negative.
    # The purpose of this bucket is to let each negative be mapped to its corresponding positive, so that the argument
    # masked in each negative is the same argument as its corresponding positive: in this way the confidence values of
    # the masked value would be directly comparable.
    posi_entries_by_sidx = {}
    for ent in posi_data_entries:
        if ent['in_partition_sidx'] not in posi_entries_by_sidx:
            posi_entries_by_sidx[ent['in_partition_sidx']] = []
        posi_entries_by_sidx[ent['in_partition_sidx']].append(ent)

    out_fp = open(args.negi_fpath, 'w', encoding='utf8')

    for ent in data_entries:
        upred, subj, obj, tsubj, tobj = parse_rel(ent)

        arg2mask = None
        for posi_ent in posi_entries_by_sidx[ent['in_partition_sidx']]:
            if posi_ent['partition_key'] != ent['partition_key']:
                continue
            posi_upred, posi_subj, posi_obj, posi_tsubj, posi_tobj = parse_rel(posi_ent)
            if posi_upred == ent['posi_upred'] and posi_subj == subj and posi_obj == obj:
                assert arg2mask is None
                if posi_ent['index_position'] == 'subj':
                    arg2mask = 'obj'
                elif posi_ent['index_position'] == 'obj':
                    arg2mask = 'subj'
                else:
                    raise AssertionError
        assert arg2mask is not None

        if arg2mask == 'subj':
            ent['answer'] = subj
            ent['answer_type'] = tsubj
            ent['index_arg'] = obj
            ent['index_position'] = 'obj'
            ent['index_type'] = tobj
        elif arg2mask == 'obj':
            ent['answer'] = obj
            ent['answer_type'] = tobj
            ent['index_arg'] = subj
            ent['index_position'] = 'subj'
            ent['index_type'] = tsubj
        else:
            raise AssertionError

        out_line = json.dumps(ent, ensure_ascii=False)
        out_fp.write(out_line+'\n')

    out_fp.close()


def find_all_matches_in_string(string, pattern):
    if len(pattern) == 0:
        print(f"Pattern with zero length!")
        return []
    id_list = []
    offset = 0
    while True:
        cur_id = string.find(pattern, offset)
        if cur_id < 0:
            break
        id_list.append(cur_id)
        offset = cur_id + len(pattern)
    return id_list


def en_find_all_matches_in_lemma_list(sent_list, pattern, lemmatizer):
    pattern = [x for x in split_str_multipat(pattern, ['_', ' ']) if len(x) > 0]

    if len(pattern) == 0:
        print(f"Pattern with zero length!")
        return []
    elif len(pattern) > 3:
        print(f"en_find_all_matches_in_lemma_list: Extremely long pattern: {pattern};")

    sent_list = [lemmatizer.lemmatize(t).lower() for t in sent_list]  # by default the lemmatization process is for a NOUN
    pattern = [lemmatizer.lemmatize(t).lower() for t in pattern]  # by default the lemmatization process is for a NOUN

    id_list = []
    offset = 0
    while offset + len(pattern) <= len(sent_list):
        mismatch_found = False
        for t, p in zip(sent_list[offset:], pattern):
            if t != p:
                mismatch_found = True
                break
        if mismatch_found:
            offset += 1
            continue
        else:
            id_list.append(offset)
            offset += len(pattern)
    return id_list


# From a given sentence, fetch the most relevant span to a given rel; if the rel is not given (None), just truncate the
# sentence to its first max_spansize tokens.
def fetch_span_by_rel(sent, rel, max_spansize, lang):
    def filter_en_sentlist(sentlist):
        new_sent_list = []
        for t in sentlist:
            if len(t) <= 2:
                new_sent_list.append(t)
            # multi-character tokens must not be all punctuations!
            else:
                all_punct = True
                for c in t:
                    if c not in string.punctuation:
                        all_punct = False
                        break
                if not all_punct:
                    new_sent_list.append(t)
                else:
                    # print(f"fetch_span_by_rel: Warning: long token all punctuation: {t};", file=sys.stderr)
                    pass
        return new_sent_list

    global lemmatizer

    if lang == 'zh':
        pass
    elif lang == 'en':
        sent = [x for x in sent.split(' ') if len(x) > 0]
        sent = filter_en_sentlist(sent)
    else:
        raise AssertionError

    if len(sent) <= max_spansize:
        return sent
    if rel is None:
        return sent[:max_spansize]

    upred, subj, obj, tsubj, tobj = parse_rel(rel)
    if lang == 'zh':
        subj_ids = find_all_matches_in_string(sent, subj)
        obj_ids = find_all_matches_in_string(sent, obj)
    elif lang == 'en':
        subj_ids = en_find_all_matches_in_lemma_list(sent, subj, lemmatizer=lemmatizer)
        obj_ids = en_find_all_matches_in_lemma_list(sent, obj, lemmatizer=lemmatizer)
    else:
        raise AssertionError

    if len(subj_ids) == 0 or len(obj_ids) == 0:
        return sent[:max_spansize]

    selected_sid = None
    selected_oid = None
    min_dist = len(sent)

    for sid in subj_ids:
        for oid in obj_ids:
            if abs(sid - oid) < min_dist:
                selected_sid = sid
                selected_oid = oid
                min_dist = abs(sid - oid)
    mid = int(selected_sid + selected_oid) // 2
    if mid < max_spansize // 2:
        return sent[:max_spansize]
    if mid > len(sent) - (max_spansize // 2):
        assert len(sent) - max_spansize > 0
        return sent[len(sent) - max_spansize:]

    assert mid - (max_spansize // 2) >= 0 and mid + (max_spansize // 2) <= len(sent)
    return sent[(mid - (max_spansize // 2)):(mid + (max_spansize // 2))]


def reconstruct_sent_from_rel(rel, max_spansize, mask_answer_flag=False, mask_token='<extra_id_0>', lang=None):
    """

    :param rel:
    :param max_spansize:
    :param mask_answer_flag:
    :param mask_token:
    :param lang:
    :return: the returned reconstructed_sent is a string for Chinese, is a list for English.
    """
    upred, subj, obj, tsubj, tobj = parse_rel(rel)

    # TODO: maybe later make this more fine-grained, like '什么', '谁', '什么时候', '哪里', (we don't have `how' here!)
    if mask_answer_flag:  # if mask_answer_flag, then mask the masked-argument with `[MASK]'
        assert 'index_position' in rel
        if rel['index_position'] == 'subj':
            obj = mask_token  # then obj should be masked;
        elif rel['index_position'] == 'obj':
            subj = mask_token  # then subj should be masked;
        else:
            raise AssertionError

    if lang == 'zh':
        assert upred[0] == '(' and upred[-1] == ')'
        upred_surface_form = upred[1:-1]
        upred_surface_form = upred_surface_form.split('.1,')
        assert len(upred_surface_form) == 2
        upred_surface_form = upred_surface_form[0]
        upred_xidxs = find_all_matches_in_string(upred_surface_form, '·X·')

        upred_charmap = []

        if len(upred_xidxs) == 0:
            upred_surface_form = upred_surface_form.replace('·', '')
            reconstructed_sent = subj + upred_surface_form + obj
            upred_charmap = [0]*len(subj) + [1]*len(upred_surface_form) + [0] * len(obj)
        elif len(upred_xidxs) == 1:
            upred_surface_form_mid, upred_surface_form_tail = upred_surface_form.split('·X·')
            upred_surface_form_mid = upred_surface_form_mid.replace('·', '')
            upred_surface_form_tail = upred_surface_form_tail.replace('·', '')
            reconstructed_sent = subj + upred_surface_form_mid + obj + upred_surface_form_tail  # we have stuck the object back into the predicate!
            upred_charmap = [0]*len(subj) + [1]*len(upred_surface_form_mid) + [0]*len(obj) + [1]*len(upred_surface_form_tail)
        else:
            raise AssertionError

    elif lang == 'en':
        # for the case of English, it's really token-map rather than charmap.
        upred_list = upred2bow(upred, lang=lang)
        subj_list = split_str_multipat(subj, [' ', '_']) if subj != mask_token else [subj]
        obj_list = split_str_multipat(obj, [' ', '_']) if obj != mask_token else [obj]
        upred_list = [x for x in upred_list if len(x) > 0]
        subj_list = [x for x in subj_list if len(x) > 0]
        obj_list = [x for x in obj_list if len(x) > 0]
        upred_charmap = [0]*len(subj_list) + [1]*len(upred_list) + [0]*len(obj_list)
        reconstructed_sent = subj_list + upred_list + obj_list
    else:
        raise AssertionError

    assert len(upred_charmap) == len(reconstructed_sent)
    if len(reconstructed_sent) > max_spansize:
        reconstructed_sent = reconstructed_sent[:max_spansize]
        upred_charmap = upred_charmap[:max_spansize]

    return reconstructed_sent, upred_charmap


def en_tokenize_and_map(tokenizer, sent_lists, word_mappings):
    sent_strs = [' '.join(slst) for slst in sent_lists]

    batch_toks = tokenizer(sent_strs, padding=True, truncation=True, max_length=512)
    batch_input_ids = batch_toks['input_ids']
    batch_token_mappings = []

    for input_ids, slst, wmap in zip(batch_input_ids, sent_lists, word_mappings):
        cur_failure_flag = False
        assert input_ids[0] in [0, 101]

        if input_ids[0] == 0:  # that means we are handling RoBERTa, which we don't have a mapping, and should fall back to using the representation at the first token.
            cur_failure_flag = True
        if len(wmap) == 0:
            cur_failure_flag = True
        perword_encoded = [tokenizer.encode(x, add_special_tokens=False) for x in slst]
        cur_tok_mapping = [0]  # this `0' is the label for the prepended [CLS] token
        tokens_ite = 1  # the first token is always 101, which is absent in perword_encoded, so we start with the second token.
        for i, word_toks in enumerate(perword_encoded):
            if cur_failure_flag:
                break
            for t in word_toks:
                if t != input_ids[tokens_ite]:
                    print(f"tokenize_and_map: Warning! Mismatch between sentential encoding and individual token encoding!", file=sys.stderr)
                    print(f"input_ids: {input_ids}", file=sys.stderr)
                    print(f"perword_encoded: {perword_encoded}", file=sys.stderr)
                    print(f"Falling back to extracting [CLS] token!", file=sys.stderr)
                    print(f"")
                    cur_failure_flag = True
                    break
                else:
                    # copy the mapping label of word i to this current token, since this token belongs to word i.
                    cur_tok_mapping.append(wmap[i])
                tokens_ite += 1

        if cur_failure_flag:
            pass
        elif input_ids[tokens_ite] != 102:
            print(f"tokenize_and_map: Warning! DIFFERENT SEQ LENGTH between sentential encoding and individual token encoding!",
                  file=sys.stderr)
            print(f"input_ids: {input_ids}", file=sys.stderr)
            print(f"perword_encoded: {perword_encoded}", file=sys.stderr)
            print(f"Falling back to extracting [CLS] token!", file=sys.stderr)
            print(f"")
            cur_failure_flag = True
        else:
            # this following pair of operations is to move behind the [SEP] token into the range of [PAD] tokens
            cur_tok_mapping.append(0)
            tokens_ite += 1
            for t in input_ids[tokens_ite:]:
                if t != 0:
                    print(
                        f"tokenize_and_map: Warning! [SEP] token not followed by [PAD]!",
                        file=sys.stderr)
                    print(f"input_ids: {input_ids}", file=sys.stderr)
                    print(f"Falling back to extracting [CLS] token!", file=sys.stderr)
                    print(f"")
                    cur_failure_flag = True
                    break
                else:
                    cur_tok_mapping.append(0)
        if cur_failure_flag:
            batch_token_mappings.append([1]+[0]*(len(input_ids)-1))
        else:
            assert len(cur_tok_mapping) == len(input_ids)
            batch_token_mappings.append(cur_tok_mapping)

    for input_ids, tok_mapping in zip(batch_input_ids, batch_token_mappings):
        assert len(input_ids) == len(tok_mapping)

    batch_token_mappings = torch.tensor(batch_token_mappings, dtype=torch.float)

    return batch_toks, batch_token_mappings


PRINT_FLAG = True


def convert_charmaps_to_tokenmaps(input_strs, input_toks, charmaps, bert_tokenizer):
    input_toks = copy.deepcopy(input_toks)
    token_maps = []
    global PRINT_FLAG

    for inbatch_id, (in_s, in_t, in_cmap) in enumerate(zip(input_strs, input_toks.input_ids, charmaps)):
        if 'roberta' in bert_tokenizer.name_or_path:
            if PRINT_FLAG is True:
                print(f"handling RoBERTa!")
                PRINT_FLAG = False
            # we fall back to taking the first token for RoBERTa!
            token_maps.append([1] + [0]*(len(in_t)-1))
            continue
        elif len(in_cmap) == 0:
            # this case means: the charmap is empty, we want to take the [CLS] token!
            token_maps.append([1] + [0]*(len(in_t)-1))
            continue
        else:
            assert len(in_s) == len(in_cmap)
        in_t = bert_tokenizer.convert_ids_to_tokens(in_t)
        in_tmap = [0] * len(in_t)
        in_t_char2tok = []
        in_t_chars = ''
        for ti, t in enumerate(in_t):
            if t in ['[CLS]', '[PAD]', '[SEP]']:
                continue
            if t[:2] == '##':
                t = t[2:]
            in_t_chars += t
            in_t_char2tok += [ti] * len(t)

        assert len(in_s) >= len(in_t_chars)
        if len(in_s) > len(in_t_chars):
            print(f"lengths of string and tokens different!")

        try:
            ite_tc = 0
            for c, m in zip(in_s, in_cmap):
                # while c != in_t_chars[ite_tc] and ite_tc < len(in_t_chars):
                # 	ite_tc += 1
                # if ite_tc >= len(in_t_chars) and ite_tc not in [' ']:
                # 	print(f"input string: {in_s}")
                # 	print(f"input tokens stringified: {in_t_chars}")
                # 	print(f"current char: {c}")
                # 	print(f"Character in input string not found!", file=sys.stderr)
                if c != in_t_chars[ite_tc]:
                    print(f"Character in original input string not found!", file=sys.stderr)
                    continue

                cur_tokid = in_t_char2tok[ite_tc]
                if m == 1:  # the token is considered as long as at least a part of this token was registered as part of the predicate.
                    in_tmap[cur_tokid] = 1
                ite_tc += 1
            if ite_tc != len(in_t_chars):
                print(f"Character in tokenized string not matched!", file=sys.stderr)
                print(f"input string: {in_s}")
                print(f"input tokens stringified: {in_t_chars}")
                print(f"last token_stringified index: {ite_tc}")
            token_maps.append(in_tmap)
        except Exception as e:
            print(f"Exception in convert_charmaps_to_tokenmaps: {e}")
            token_maps.append([1] + [0]*(len(in_t)-1))

    token_maps = torch.tensor(token_maps, dtype=torch.float)
    assert len(token_maps.shape) == 2
    assert token_maps.shape[0] == len(input_toks.input_ids) and token_maps.shape[1] == len(input_toks.input_ids[0])
    return token_maps


# This preprocessing is only for Chinese, where there are no explicit word separators, but word separators are needed for
# mT5 to understand the sentence.
def prepare_string_for_T5Tokenizer(sent):
    new_sent = ''
    last_word_is_ascii = False
    for char in sent:
        if char.isascii():
            new_sent += char
            last_word_is_ascii = True
        else:
            if last_word_is_ascii is True:
                new_sent += ' '

            new_sent += char + ' '
            last_word_is_ascii = False
    if not new_sent.endswith(' '):
        new_sent += ' '
    return new_sent


# TODO: for Chinese version we use a random cutoff to further reduce computational cost!
def calc_per_entry_score_ss_ssc(query_ent: dict, all_ref_rels: list, max_spansize: int, method: str, ss_model, ss_data_preprocer,
                            bert_device, max_context_size=None, debug=False, is_wh=False, batch_size=16, lang=None,
                                data_parallel: bool = False):
    if is_wh is True:
        raise NotImplementedError
    else:
        pass

    # for wh-tasks, it feels like this masked answer should be replaced by the candidate answers?
    query_sent, _ = reconstruct_sent_from_rel(query_ent, max_spansize, mask_answer_flag=False, mask_token=None, lang=lang)

    if lang == 'en':
        if max_context_size is not None and len(all_ref_rels) > max_context_size:
            print(f"maximum context size exceeded! Sampling {max_context_size} contexts out of {len(all_ref_rels)}!")
            sample_ref_rels = random.sample(all_ref_rels, k=max_context_size)
        else:
            sample_ref_rels = all_ref_rels
    elif lang == 'zh':
        sample_ref_rels = all_ref_rels
        if len(all_ref_rels) > 90:
            sample_ref_rels = random.sample(all_ref_rels[:500], k=90)
        else:
            pass
        # elif len(all_ref_rels) > 200:
        #     rho = random.random()
        #     sample_ref_rels = random.sample(all_ref_rels, k=200) if rho < 1 else all_ref_rels
        if len(all_ref_rels) > 60:
            rho = random.random()
            sample_ref_rels = random.sample(all_ref_rels[:500], k=60) if rho < 0.7 else sample_ref_rels
        else:
            pass
        if len(all_ref_rels) > 30:
            rho = random.random()
            sample_ref_rels = random.sample(all_ref_rels[:500], k=30) if rho < 0.3 else sample_ref_rels
        else:
            pass
        assert len(sample_ref_rels) <= 150

    else:
        raise AssertionError

    assert len(all_ref_rels) > 0
    ref_inputstrs = []
    ref_answers = []
    ref_scores = []

    for rrel in sample_ref_rels:
        if is_wh is True:
            _, rsubj, robj, _, _ = parse_rel(rrel)
            if rsubj == query_ent['index_arg']:
                rans = robj
            elif robj == query_ent['index_arg']:
                rans = rsubj
            else:
                raise AssertionError
            ref_answers.append(rans)
        else:
            pass

        r_inputstr, _ = reconstruct_sent_from_rel(rrel, max_spansize, mask_answer_flag=False, mask_token=None, lang=lang)
        ref_inputstrs.append(r_inputstr)

    assert len(ref_inputstrs) > 0

    ref_chunks = []
    offset = 0  # this ``offset'' is the starting point of each chunk, so the last chunk is also included!
    while offset < len(ref_inputstrs):
        ref_chunks.append((offset, min(offset + batch_size, len(ref_inputstrs))))
        offset += batch_size

    for chunk in ref_chunks:
        torch.cuda.empty_cache()
        if chunk[1] == chunk[0]:
            continue
        # the reference should entail the query, not vice versa.
        cur_chunk_pairs = [(r_inputstr, query_sent) for r_inputstr in ref_inputstrs[chunk[0]:chunk[1]]]

        cur_chunk_instances = []
        for prem, hypo in cur_chunk_pairs:
            inst = ss_data_preprocer.create_instances((' '.join(prem), '', ''), (' '.join(hypo), '', ''), False, language=lang.upper())

            assert len(inst) == 1
            inst = inst[0]
            cur_chunk_instances.append((inst[SENT_KEY], inst[ANTI_KEY], inst[LABEL_KEY]))

        if data_parallel is True:
            cur_chunk_instances = ss_model.module.collate(cur_chunk_instances)
        elif data_parallel is False:
            cur_chunk_instances = ss_model.collate(cur_chunk_instances)
        else:
            raise AssertionError
        cur_chunk_instances = ([x.to(bert_device) for x in cur_chunk_instances[0]],
                               [x.to(bert_device) for x in cur_chunk_instances[1]],
                               cur_chunk_instances[2].to(bert_device))

        if data_parallel is True:
            res_dict = ss_model.module.validation_step(cur_chunk_instances, 0)
        elif data_parallel is False:
            res_dict = ss_model.validation_step(cur_chunk_instances, 0)
        else:
            raise AssertionError

        assert torch.all((res_dict['scores'] > -1) & (res_dict['scores'] < 1))
        res_dict['scores'] = (res_dict['scores'] + 1) / 2
        assert len(res_dict['scores'].shape) == 1
        ref_scores.append(res_dict['scores'].detach().cpu())

    ref_scores = torch.cat(ref_scores)

    assert len(ref_scores.shape) == 1 and ref_scores.shape[0] == len(ref_inputstrs)

    if is_wh:
        raise NotImplementedError
    else:
        if len(ref_inputstrs) > 0:
            return_value = torch.max(ref_scores).item()
            cur_argmax = torch.argmax(ref_scores).item()
            if debug:
                print(f"cur sims shape: {ref_scores.shape}; best value: {return_value};")
                print(f"query rel: {query_ent['r']}")
                print(f"best ref rel: {sample_ref_rels[cur_argmax]['r']}")
        else:
            return_value = 0.0
            if debug:
                print(f"No relevant relations found!")

    return return_value


# This function calculates the maximum cosine similarity score.
def calc_per_entry_score_bert(query_ent, ref_rels, ref_sents, method, max_spansize, bert_model, bert_tokenizer,
                              bert_device, max_context_size=None, debug=False, is_wh=False, batch_size=64, lang=None):
    # print(f"Is_wh equals to {is_wh};")
    torch.cuda.empty_cache()
    assert method in ['bert1A', 'bert2A', 'bert3A']
    assert isinstance(bert_model, transformers.BertModel) or isinstance(bert_model, transformers.RobertaModel)
    # the data entry can be positive or negative, so that must be scenario 3; but the references here can be scenario 2
    query_sent, query_charmap = reconstruct_sent_from_rel(query_ent, max_spansize, mask_answer_flag=is_wh,
                                                          mask_token='[MASK]', lang=lang)

    if lang == 'zh':
        query_toks = bert_tokenizer([query_sent], padding=True)
        query_tokenmap = convert_charmaps_to_tokenmaps([query_sent], query_toks, [query_charmap], bert_tokenizer)
    elif lang == 'en':
        query_toks, query_tokenmap = en_tokenize_and_map(tokenizer=bert_tokenizer, sent_lists=[query_sent],
                                                         word_mappings=[query_charmap])
    else:
        raise AssertionError
    query_toks = query_toks.convert_to_tensors('pt')
    query_toks = query_toks.to(bert_device)
    query_outputs = bert_model(input_ids=query_toks.input_ids)
    query_vecs = query_outputs.last_hidden_state
    if method in ['bert1A', 'bert2A']:
        query_vecs = query_vecs[:, 0, :].cpu().detach().numpy()
    elif method in ['bert3A']:
        assert query_vecs.shape[0] == 1
        assert len(query_vecs.shape) == 3 and len(query_tokenmap.shape) == 2 and \
               query_vecs.shape[0] == query_tokenmap.shape[0] and query_vecs.shape[1] == query_tokenmap.shape[1]

        # query_tokenmap is a mask tensor where only the positions corresponding to predicates are allowed.
        query_tokenmap = query_tokenmap.view(query_vecs.shape[0], query_vecs.shape[1], 1).expand(-1, -1, query_vecs.shape[-1])
        query_vecs = query_vecs.cpu()
        query_vecs *= query_tokenmap  # [batch_size, seq_length, dimensions]
        query_vecs = query_vecs.mean(dim=1).detach().numpy()
    else:
        raise AssertionError

    if max_context_size is not None and len(ref_sents) > max_context_size:
        if method in ['bert1A']:
            print(f"bert1A context having extraordinarily many sentences! {len(ref_sents)} with threshold at {max_context_size};")
        # if debug:
        print(f"maximum context size exceeded! Sampling {max_context_size} contexts out of {len(ref_sents)}!")
        assert ref_rels is None or len(ref_rels) == len(ref_sents)
        sample_ids = random.sample(range(len(ref_sents)), k=max_context_size)
        # TODO: note that there is a bug here: so far in the results, still the full versions of ref_rels and ref_sents are used;
        # TODO: this is true for the entGraph methods as well as bert2 and bert3
        new_ref_rels = [] if ref_rels is not None else None
        new_ref_sents = []
        for i in sample_ids:
            new_ref_sents.append(ref_sents[i])
            if ref_rels is not None:
                new_ref_rels.append(ref_rels[i])
        ref_rels = new_ref_rels
        ref_sents = new_ref_sents

    ref_emb_inputstrs = []
    ref_emb_charmaps = []
    ref_answers = []  # this list is populated only when is_wh is True!
    ref_emb_outputvecs = []
    if method in ['bert2A', 'bert3A']:
        assert len(ref_rels) == len(ref_sents)
        for rrel, rsent in zip(ref_rels, ref_sents):

            if is_wh is True:
                _, rsubj, robj, _, _ = parse_rel(rrel)
                if rsubj == query_ent['index_arg']:
                    rans = robj  # the prediction
                elif robj == query_ent['index_arg']:
                    rans = rsubj  # the prediction
                else:
                    raise AssertionError
                ref_answers.append(rans)
            else:
                pass

            if method == 'bert2A':
                r_inputstr = fetch_span_by_rel(rsent, rrel, max_spansize=max_spansize, lang=lang)
                r_charmap = []
                ref_emb_inputstrs.append(r_inputstr)
                ref_emb_charmaps.append(r_charmap)
            elif method == 'bert3A':
                r_inputstr, r_charmap = reconstruct_sent_from_rel(rrel, max_spansize, lang=lang)
                ref_emb_inputstrs.append(r_inputstr)
                ref_emb_charmaps.append(r_charmap)
            else:
                raise AssertionError
    else:
        assert method == 'bert1A' and is_wh is False
        for rsent in ref_sents:
            r_inputstr = fetch_span_by_rel(rsent, None, max_spansize=max_spansize, lang=lang)
            r_charmap = []
            ref_emb_inputstrs.append(r_inputstr)
            ref_emb_charmaps.append(r_charmap)

    # if is_wh:
    # 	assert isinstance(bert_tokenizer, transformers.T5Tokenizer)
    # 	new_ref_emb_inputstrs = []
    # 	for string in ref_emb_inputstrs:
    # 		new_string = prepare_string_for_T5Tokenizer(string)
    # 		new_ref_emb_inputstrs.append(new_string)
    # 	ref_emb_inputstrs = new_ref_emb_inputstrs

    # raise NotImplementedError(f"charmap is not converted to tokenmap or taken into account!")

    ref_emb_chunks = []
    offset = 0  # this ``offset'' is the starting point of each chunk, so the last chunk is also included!
    while offset < len(ref_emb_inputstrs):
        ref_emb_chunks.append((offset, min(offset + batch_size, len(ref_emb_inputstrs))))
        offset += batch_size

    for chunk in ref_emb_chunks:
        if chunk[1] == chunk[0]:  # do not attempt to send empty input into the model!
            continue
        if lang == 'zh':
            ref_emb_inputtoks = bert_tokenizer(ref_emb_inputstrs[chunk[0]:chunk[1]], padding=True, truncation=True, max_length=max_spansize)
            ref_emb_tokenmaps = convert_charmaps_to_tokenmaps(ref_emb_inputstrs[chunk[0]:chunk[1]], ref_emb_inputtoks,
                                                              ref_emb_charmaps[chunk[0]:chunk[1]], bert_tokenizer)
        elif lang == 'en':
            ref_emb_inputtoks, ref_emb_tokenmaps = en_tokenize_and_map(tokenizer=bert_tokenizer,
                                                                       sent_lists=ref_emb_inputstrs[chunk[0]:chunk[1]],
                                                                       word_mappings=ref_emb_charmaps[chunk[0]:chunk[1]])
        else:
            raise AssertionError
        ref_emb_inputtoks = ref_emb_inputtoks.convert_to_tensors('pt')
        ref_emb_inputtoks = ref_emb_inputtoks.to(bert_device)
        try:
            ref_encoder_outputs = bert_model(**ref_emb_inputtoks)
        except Exception as e:
            print(f"{e}")
            print(f"ref_emb_inputstrs: ")
            print(f"{ref_emb_inputstrs}")
            print(f"ref_emb_inputtoks: ")
            print(f"{ref_emb_inputtoks}")
            raise
        ref_encoder_outputs = ref_encoder_outputs.last_hidden_state
        # ref_emb_tokenmap is a mask tensor where only the positions corresponding to predicates are allowed.
        ref_emb_tokenmaps = ref_emb_tokenmaps.view(ref_encoder_outputs.shape[0], ref_encoder_outputs.shape[1], 1).expand(-1, -1, ref_encoder_outputs.shape[-1])
        ref_encoder_outputs = ref_encoder_outputs.cpu()
        ref_encoder_outputs *= ref_emb_tokenmaps  # [batch_size, seq_length, dimensions]
        ref_encoder_outputs = ref_encoder_outputs.mean(dim=1).detach().numpy()
        for bidx in range(ref_encoder_outputs.shape[0]):
            ref_emb_outputvecs.append(ref_encoder_outputs[bidx, :])
    assert len(ref_emb_outputvecs) == len(ref_emb_inputstrs)
    ref_emb_outputvecs = np.array(ref_emb_outputvecs)

    # for wh-questions, the returned value is a dict, with answers as keys and corresponding similarities as values;
    # for boolean questions, the returned value is a float number, which is the maximum score.
    if is_wh:
        assert method not in ['bert1A']
        if len(ref_emb_inputstrs) > 0:
            cur_sims = calc_simscore(query_vecs, ref_emb_outputvecs)
            assert len(cur_sims.shape) == 2 and cur_sims.shape[0] == 1
            cur_sims = cur_sims[0].tolist()
            assert len(cur_sims) == len(ref_answers)
            return_value = {}

            for s, a in zip(cur_sims, ref_answers):
                if a not in return_value:
                    return_value[a] = s
                elif s > return_value[a]:
                    return_value[a] = s
                else:
                    pass
            return_value = {a: s for (a, s) in sorted(return_value.items(), key=lambda x: x[1], reverse=True)[:50]}

            if debug:
                print(f"cur sims shape: {len(cur_sims)}")
                print(f"query rel: {query_ent['r']}")
                print(f"best answers: ")
                print_count = 0
                for a in return_value:  # print the top 10 answers (unique)
                    print(f"{a}: {return_value[a]};")
                    print_count += 1
                    if print_count >= 10:
                        break
        else:
            return_value = {}
            if debug:
                print(f"No relevant relations found!")
    else:
        if len(ref_emb_inputstrs) > 0:
            cur_sims = calc_simscore(query_vecs, ref_emb_outputvecs)
            assert len(cur_sims.shape) == 2 and cur_sims.shape[0] == 1
            return_value = np.amax(cur_sims)
            cur_argmax_sim = np.argmax(cur_sims)
            if debug:
                print(f"cur sims shape: {cur_sims.shape}; best value: {return_value};")
                print(f"query rel: {query_ent['r']}")
                if ref_rels is not None:
                    print(f"best ref rel: {ref_rels[cur_argmax_sim]['r']}")
                else:
                    print(f"Best ref sent: {ref_sents[cur_argmax_sim]}")
        else:
            return_value = 0.0
            if debug:
                print(f"No relevant relations found!")
    return return_value


def calc_per_entry_score_T5(query_ent, ref_rels, ref_sents, method, max_spansize, bert_model, bert_tokenizer,
                            max_context_size, debug, batch_size, lang):
    def sort_output_seqs(decoded_list, effective_batch_size):
        assert len(decoded_list) == effective_batch_size * num_return_sentences
        output_seqs = []
        i = 0
        for i, x in enumerate(decoded_list):
            if i % num_return_sentences == 0:
                output_seqs.append([])
            output_seqs[-1].append(x)
        assert i % num_return_sentences == (num_return_sentences-1)
        assert len(output_seqs) == effective_batch_size
        return output_seqs

    # The final scores are computed as the highest score for yes/Yes deducting the highest score for no/No; if either
    # yes or no is missing from the sequence, skip the entry; the scores goes through (+1)/2 to be normalized to [0, 1]
    def get_scores_from_seqs(output_seqs, seq_scores):
        final_scores = []
        assert len(output_seqs) == seq_scores.shape[0]
        for o_seq, o_score in zip(output_seqs, seq_scores):
            yes_scr = None
            no_scr = None
            assert len(o_seq) == o_score.shape[0]
            for string, scr in zip(o_seq, o_score):
                if string in ['yes', 'Yes', 'yes;', 'Yes;'] and yes_scr is None:
                    yes_scr = float(scr.cpu())
                elif string in ['no', 'No', 'no;', 'No;'] and no_scr is None:
                    no_scr = float(scr.cpu())
            if yes_scr is None or no_scr is None:
                final_scores.append(0.0)
            else:
                assert -1 < yes_scr - no_scr < 1
                final_scores.append((yes_scr-no_scr+1)/2)
        return final_scores

    example_str = 'A drug kills infections, does this mean a drug is useful in infections? Yes ; ' \
              'a drug is useful in infections, does this mean a drug kills infections? No ;'
    num_beams = 5
    num_return_sentences = 5

    torch.cuda.empty_cache()
    assert method in ['T51A', 'T53A']
    if method in ['T51A']:
        assert ref_sents is not None and ref_rels is None
    elif method in ['T53A']:
        assert ref_rels is not None and ref_sents is None
    else:
        raise AssertionError

    if lang == 'en':
        pass
    elif lang == 'zh':
        raise NotImplementedError
    else:
        raise AssertionError

    assert isinstance(bert_model, transformers.T5ForConditionalGeneration)
    query_sent, query_charmap = reconstruct_sent_from_rel(query_ent, max_spansize, mask_answer_flag=False, lang=lang)
    assert isinstance(query_sent,list)
    query_sent = ' '.join(query_sent)

    if max_context_size is not None:
        if ref_rels is not None and len(ref_rels) > max_context_size:
            print(f"maximum context size exceeded! Sampling {max_context_size} contexts out of {len(ref_rels)}!")
            sample_ids = random.sample(range(len(ref_rels)), k=max_context_size)
            new_ref_rels = []
            for i in sample_ids:
                new_ref_rels.append(ref_rels[i])
            ref_rels = new_ref_rels
        elif ref_sents is not None and len(ref_sents) > max_context_size:
            print(f"maximum context size exceeded! Sampling {max_context_size} contexts out of {len(ref_sents)}!")
            sample_ids = random.sample(range(len(ref_sents)), k=max_context_size)
            new_ref_sents = []
            for i in sample_ids:
                new_ref_sents.append(ref_sents[i])
            ref_sents = new_ref_sents
        else:
            pass

    ref_emb_inputstrs = []

    if method in ['T53A']:
        for rrel in ref_rels:
            r_inputstr, r_charmap = reconstruct_sent_from_rel(rrel, max_spansize, lang=lang)
            assert isinstance(r_inputstr, list)
            r_inputstr = ' '.join(r_inputstr)
            input_template = f"{example_str} {r_inputstr}, does this mean {query_sent}? <extra_id_0>"
            ref_emb_inputstrs.append(input_template)
    elif method in ['T51A']:
        for rsent in ref_sents:
            r_inputstr = fetch_span_by_rel(rsent, None, max_spansize=max_spansize, lang=lang)
            assert isinstance(r_inputstr, list)
            r_inputstr = ' '.join(r_inputstr)
            input_template = f"{example_str} {r_inputstr}, does this mean {query_sent}? <extra_id_0>"
            ref_emb_inputstrs.append(input_template)
    else:
        raise AssertionError

    ref_emb_chunks = []
    offset = 0
    while offset < len(ref_emb_inputstrs):
        ref_emb_chunks.append((offset, min(offset + batch_size, len(ref_emb_inputstrs))))
        offset += batch_size

    return_value = 0.0
    chosen_ref = None
    for chunk in ref_emb_chunks:
        if chunk[1] == chunk[0]:
            continue
        cur_chunk_strs = ref_emb_inputstrs[chunk[0]:chunk[1]]
        input_encoded = bert_tokenizer(cur_chunk_strs, add_special_tokens=True,
                                       return_tensors='pt', padding=True)
        if torch.cuda.is_available():
            input_ids = input_encoded['input_ids'].to('cuda:0')
            attention_mask = input_encoded['attention_mask'].to('cuda:0')
        else:
            input_ids = input_encoded['input_ids']
            attention_mask = input_encoded['attention_mask']
        outputs = bert_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams,
                                      num_return_sequences=num_return_sentences, max_length=3, return_dict_in_generate=True,
                                      output_scores=True)
        exp_sequence_scores = torch.exp(outputs.sequences_scores)
        exp_sequence_scores = exp_sequence_scores.view(-1, num_return_sentences)
        decoded = bert_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        output_sequences = sort_output_seqs(decoded, exp_sequence_scores.shape[0])
        calced_scores = get_scores_from_seqs(output_sequences, exp_sequence_scores)
        for sidx, scr in enumerate(calced_scores):
            assert 0 <= scr <= 1
            if scr > return_value:
                return_value = scr
                chosen_ref = ref_emb_inputstrs[sidx]

    if return_value == 0.0:
        assert chosen_ref is None
        if debug:
            print(f"No relevant relations found!")
    if chosen_ref is not None and debug:
        print(f"best ref: {chosen_ref};")

    # return_value = float(return_value.cpu())
    if debug:
        print(f"return value: {return_value}")
    return return_value


# This function prepends the context to the question, and outputs softmaxed logits.
def in_context_prediction_bert(query_ent, ref_rels, ref_sents, method, max_spansize, bert_model, bert_tokenizer,
                              bert_device, max_seqlength=600, debug=False, is_wh=False, lang=None):
    if lang == 'zh':
        pass
    elif lang == 'en':
        print(f"in_context_prediction_bert not yet compatible with English!", file=sys.stderr)
        raise NotImplementedError
    else:
        raise AssertionError
    assert is_wh is True and isinstance(bert_tokenizer, transformers.T5Tokenizer)
    assert method in ['bert1B', 'bert2B', 'bert3B']
    # the data entry can be positive or negative, so that must be scenario 3; but the references here can be scenario 2
    query_sent, _ = reconstruct_sent_from_rel(query_ent, max_spansize, mask_answer_flag=is_wh, lang=lang)
    total_length = len(query_sent)

    concat_sent = ''

    ref_emb_inputstrs = []
    if method in ['bert2B', 'bert3B']:
        assert len(ref_rels) == len(ref_sents)
        for rrel, rsent in zip(ref_rels, ref_sents):
            if method == 'bert2B':
                ref_emb_inputstrs.append(fetch_span_by_rel(rsent, rrel, max_spansize=max_spansize))
            elif method == 'bert3B':
                r_inputstr, _ = reconstruct_sent_from_rel(rrel, max_spansize, lang=lang)
                ref_emb_inputstrs.append(r_inputstr)
            else:
                raise AssertionError

        random.shuffle(ref_emb_inputstrs)
        for rsent in ref_emb_inputstrs:
            # this total length includes the length of the query, and is not the same as len(concat_sent) right now
            # (in which the query sent has not been added)
            if total_length + len(rsent) < max_seqlength:
                concat_sent += rsent
                total_length += len(rsent)
                if rsent.endswith('。'):
                    pass
                else:
                    concat_sent += '。'
                    total_length += 1

    else:
        for rsent in ref_sents:
            if method == 'bert1B':
                ref_emb_inputstrs.append(fetch_span_by_rel(rsent, None, max_spansize=max_spansize))
            else:
                raise AssertionError

        # Don't shuffle in this setting: we need the articles to stay together, with the most relevant (by tf-idf) at top!
        for rsent in ref_emb_inputstrs:
            # break if the next sentence cannot fit in to max_seqlength: we need consecutive sentences in this setting!
            if total_length + len(rsent) >= max_seqlength:
                break
            concat_sent += rsent
            total_length += len(rsent)
            if rsent.endswith('。'):
                pass
            else:
                concat_sent += '。'
                total_length += 1

    concat_sent += query_sent
    assert len(concat_sent) < max_seqlength + 10
    concat_sent = prepare_string_for_T5Tokenizer(concat_sent)

    query_toks = bert_tokenizer.encode_plus(concat_sent, add_special_tokens=True, return_tensors='pt')
    input_ids = query_toks['input_ids'].to(bert_device)
    query_outputs = bert_model.generate(input_ids=input_ids, num_beams=200, num_return_sequences=50, max_length=20,
                                        return_dict_in_generate=True, output_scores=True)
    exp_sequences_scores = torch.exp(query_outputs.sequences_scores)

    end_tokens = ['</s>', '<extra_id_1>']
    return_values = {}
    assert query_outputs.sequences.shape[0] == exp_sequences_scores.shape[0]
    for aidx, (ans, ans_scr) in enumerate(zip(query_outputs.sequences, exp_sequences_scores)):
        ans = bert_tokenizer.decode(ans[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # enumerate all end_tokens, chunk the answer with the shortest one.
        for end_token in end_tokens:
            if end_token in ans:
                _end_token_idx = ans.index(end_token)
                ans = ans[:_end_token_idx]
            else:
                pass
        ans = ans.replace(' ', '')
        if ans not in return_values:
            return_values[ans] = 0
        if return_values[ans] < ans_scr:
            assert return_values[ans] == 0
            return_values[ans] = ans_scr

    return return_values


def find_samestr_scores(r_tpred: str, q_tpred: str, all_predstr_to_upreds: dict, _graph, effective_featid: int,
                        reduce: str, lang: str, threshold_samestr: bool, debug: bool):
    assert reduce in ['max', 'avg']
    assert lang == 'en'
    try:
        r_tplst = r_tpred.split('#')
        q_tplst = q_tpred.split('#')
        assert len(r_tplst) >= 3 and len(q_tplst) >= 3
        r_upred = '#'.join(r_tplst[:-2])
        r_tsubj, r_tobj = r_tplst[-2:]
        q_upred = '#'.join(q_tplst[:-2])
        q_tsubj, q_tobj = q_tplst[-2:]
    except Exception as e:
        print(e)
        print(r_tpred)
        print(q_tpred)
        raise
    _, r_upredstr = rel2concise_str(r_upred, '', '', '', '', lang=lang)
    _, q_upredstr = rel2concise_str(q_upred, '', '', '', '', lang=lang)
    assert (r_tsubj == q_tsubj and r_tobj == q_tobj) or (r_tsubj == q_tobj and r_tobj == q_tsubj), \
        f"Assertion error: r_tsubj: {r_tsubj}; r_tobj: {r_tobj}; q_tsubj: {q_tsubj}; q_tobj: {q_tobj}"

    q_tps = '#'.join([q_tsubj, q_tobj])
    r_tps = '#'.join([r_tsubj, r_tobj])
    q_nounderscore_tps = q_tps.replace('_1', '').replace('_2', '')
    r_nounderscore_tps = r_tps.replace('_1', '').replace('_2', '')

    if (q_nounderscore_tps not in all_predstr_to_upreds or r_nounderscore_tps not in all_predstr_to_upreds) and debug:
        print(f"find_samestr_scores: {q_nounderscore_tps} or {r_nounderscore_tps} is missing from "
              f"``all_predstr_to_upreds''", file=sys.stderr)

    if q_nounderscore_tps in all_predstr_to_upreds and q_upredstr in all_predstr_to_upreds[q_nounderscore_tps]:
        q_samestr_upreds = all_predstr_to_upreds[q_nounderscore_tps][q_upredstr]['p']
        if len(q_samestr_upreds) > 20 and debug:
            print(f"find_entailment_matches_from_graph: q_samestr_upreds length exceeding 20: {len(q_samestr_upreds)}")
            print(f"q_samestr_upreds: {q_samestr_upreds}")
        q_samestr_upreds = q_samestr_upreds[:20]
    else:
        q_samestr_upreds = []
    if r_nounderscore_tps in all_predstr_to_upreds and r_upredstr in all_predstr_to_upreds[r_nounderscore_tps]:
        r_samestr_upreds = all_predstr_to_upreds[r_nounderscore_tps][r_upredstr]['p']
        if len(r_samestr_upreds) > 20 and debug:
            print(f"find_entailment_matches_from_graph: r_samestr_upreds length exceeding 20: {len(r_samestr_upreds)}")
            print(f"r_samestr_upreds: {r_samestr_upreds}")
        r_samestr_upreds = r_samestr_upreds[:20]
    else:
        r_samestr_upreds = []

    samestr_all_scores = []
    samestr_all_rtpreds = []

    for r_samestr_up, r_ssup_occ in r_samestr_upreds:
        for q_samestr_up, q_ssup_occ in q_samestr_upreds:
            if threshold_samestr and (r_ssup_occ < 3 or q_ssup_occ < 3):
                continue
            r_samestr_tp = '#'.join([r_samestr_up, r_tps])
            q_samestr_tp = '#'.join([q_samestr_up, q_tps])
            cur_samestr_tscrs = _graph.get_features(r_samestr_tp, q_samestr_tp)

            if cur_samestr_tscrs is not None:
                curfeat_samestr_tscr = cur_samestr_tscrs[effective_featid]
                samestr_all_scores.append(curfeat_samestr_tscr)
                samestr_all_rtpreds.append(r_samestr_tp)

    if len(samestr_all_scores) == 0:
        samestr_final_scr = None
        samestr_final_rtpred = None
    else:
        if reduce == 'max':
            samestr_final_scr, samestr_final_rtpred = max(zip(samestr_all_scores, samestr_all_rtpreds), key=lambda x: x[0])
        elif reduce == 'avg':
            samestr_final_scr = sum(samestr_all_scores) / len(samestr_all_scores)
            samestr_final_rtpred = r_tpred
        else:
            raise AssertionError

    return samestr_final_scr, samestr_final_rtpred


def get_fuzzy_score_for_pair(_graph, r_tpred, q_tpred, all_predstr_to_upreds, effective_feat_idx, lang,
                             threshold_samestr, debug):
    is_direct_true_ent_flag = False
    is_samestr_true_ent_flag = False
    res_tscore_ref = None  # This is not necessarily the r_tpred, it could also be its fuzzy match.
    res_tscore = None

    cur_tscores = _graph.get_features(r_tpred, q_tpred)
    if cur_tscores is not None:
        cur_tscores = [x for x in cur_tscores]
        # print(f"cur tscores length: {len(cur_tscores)}")
        # print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
        if 0 < cur_tscores[0] < 0.99:
            is_direct_true_ent_flag = True
        curfeat_cur_tscore = cur_tscores[effective_feat_idx]
        res_tscore_ref = r_tpred
        res_tscore = curfeat_cur_tscore
    else:
        curfeat_cur_tscore = None

    # the last condition means that we do back off to other upreds with the same pred_str
    if (curfeat_cur_tscore is None or curfeat_cur_tscore == 0) and all_predstr_to_upreds is not None:
        curfeat_samestr_tscore, maxi_ref_tpred = find_samestr_scores(r_tpred, q_tpred,
                                                                     all_predstr_to_upreds, _graph,
                                                                     effective_feat_idx, reduce='max', lang=lang,
                                                                     threshold_samestr=threshold_samestr,
                                                                     debug=debug)
        if curfeat_samestr_tscore is not None:
            assert maxi_ref_tpred is not None
            if 0 < curfeat_samestr_tscore < 0.99:
                is_samestr_true_ent_flag = True

            res_tscore_ref = maxi_ref_tpred
            res_tscore = curfeat_samestr_tscore

    assert res_tscore is not None or res_tscore_ref is None
    return res_tscore, res_tscore_ref, is_direct_true_ent_flag, is_samestr_true_ent_flag


# This ``aligned'' is valid only when the premise is backup-smoothed so we need to alter the types for the hypothesis (the query)
# When altering the types of the premise (the reference), they are always ``aligned'' because they are the standard.
def get_backuptype_tpred(tpred, aligned=True):
    upred = tpred.split('#')[0]
    # regardless of what order the smoothed premises have, the hypothesis should always be thing_1#thing_2 when ``aligned'' and
    # thing_2#thing_1 otherwise.
    if aligned is True:
        return '#'.join([upred, 'thing_1', 'thing_2'])
    elif aligned is False:
        return '#'.join([upred, 'thing_2', 'thing_1'])
    else:
        raise AssertionError


def find_entailment_matches_from_graph(_graph, _backup_graph, _gr_predstr2preds, _bkupgr_predstr2preds, graph_deducer,
                                       ent, ref_rels, typematch_flag, feat_idx, all_predstr_to_upreds,
                                       debug: bool = False, lang: str = '', threshold_samestr: bool = False,
                                       smooth_p: str = None, smooth_h: str = None, smooth_sim_order: float = 1.0,
                                       smoothing_k: int = 4,
                                       available_graph_types: set = None, num_prems_to_check: int = None):

    assert lang in ['en', 'zh']

    # find entailment matches for one entry from one graph
    maximum_tscore = None  # typed score
    maximum_uscore = None  # BackOffAvg
    maximum_tsscore = None  # Smoothing with LM for the original graph
    maximum_usscore = None
    max_tscore_ref = None
    max_uscore_ref = None
    max_tsscore_ref = None
    max_usscore_ref = None
    prm_all_missing_flag = None  # these two flags have three states: True, False, Unset (None)
    hyp_missing_flag = None

    q_upred, q_subj, q_obj, q_tsubj, q_tobj = parse_rel(ent)
    qup_halves = q_upred[q_upred.find('(') + 1:q_upred.find(')')].split(',')
    qup_left = qup_halves[0].split('.')
    if qup_left[-1].isdigit():
        qup_left = qup_left[:-1]
    else:
        print(f"``qup_left'' does not end with digits: {qup_left}!")
    if len(qup_left) == 0:
        return None, None, None, None, 0, 0, 0, False, False
    assert '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
    if q_tsubj == q_tobj:
        q_tsubj = q_tsubj + '_1'
        q_tobj = q_tobj + '_2'
    q_tpred_querytype = '#'.join([q_upred, q_tsubj, q_tobj])
    assert len(_graph.types) == 2
    q_tpred_graphtype_fwd = '#'.join([q_upred, _graph.types[0], _graph.types[1]])
    q_tpred_graphtype_rev = '#'.join([q_upred, _graph.types[1], _graph.types[0]])

    q_nnode_bkup_gr_flag = None
    if q_tpred_querytype in _graph.pred2Node:
        q_in_graph = True
        hyp_missing_flag = False
    else:
        q_in_graph = False
        hyp_missing_flag = True

    if smooth_h is not None:
        if smooth_h == 'lm':
            if len(q_tpred_querytype.split('#')) == 3:
                assert len(q_tpred_graphtype_fwd.split('#')) == 3
                assert len(q_tpred_graphtype_rev.split('#')) == 3
                try:
                    if typematch_flag is True:
                        q_nearest_nodes = graph_deducer.get_nearest_node(preds=[q_tpred_querytype], k=smoothing_k,
                                                                         available_graphs=available_graph_types, target_typing=None,
                                                                         ablated_preds=[])
                        q_nearest_unodes_fwd = None
                        q_nearest_unodes_rev = None
                        example_nnode_tps = q_nearest_nodes[0][0][0].split('#')[1:]
                        if (example_nnode_tps[0] == q_tsubj and example_nnode_tps[1] == q_tobj) or \
                                (example_nnode_tps[0] == q_tobj and example_nnode_tps[1] == q_tsubj):
                            q_nnode_bkup_gr_flag = False
                        else:
                            assert all(x in ['thing_1', 'thing_2'] for x in example_nnode_tps)
                            q_nnode_bkup_gr_flag = True
                    elif typematch_flag is False:
                        q_nearest_nodes = None
                        q_nnode_bkup_gr_flag = None
                        q_nearest_unodes_fwd = None
                        q_nearest_unodes_rev = None
                        q_nnode_bkup_gr_flag_fwd = None
                        q_nnode_bkup_gr_flag_rev = None
                        # q_nearest_unodes_fwd = graph_deducer.get_nearest_node(preds=[q_tpred_graphtype_fwd], k=smoothing_k,
                        #                                                  available_graphs=available_graph_types,
                        #                                                  target_typing=None,
                        #                                                  ablated_preds=[])
                        # q_nearest_unodes_rev = graph_deducer.get_nearest_node(preds=[q_tpred_graphtype_rev], k=smoothing_k,
                        #                                                  available_graphs=available_graph_types,
                        #                                                  target_typing=None,
                        #                                                  ablated_preds=[])
                        # q_nearest_nodes = None
                        # example_nnode_tps_fwd = q_nearest_unodes_fwd[0][0][0].split('#')[1:]
                        # example_nnode_tps_rev = q_nearest_unodes_rev[0][0][0].split('#')[1:]
                        # if (example_nnode_tps_fwd[0] == _graph.types[0] and example_nnode_tps_fwd[1] == _graph.types[1]) or \
                        #         (example_nnode_tps_fwd[0] == _graph.types[1] and example_nnode_tps_fwd[1] == _graph.types[0]):
                        #     q_nnode_bkup_gr_flag_fwd = False
                        # else:
                        #     assert all(x in ['thing_1', 'thing_2'] for x in example_nnode_tps_fwd)
                        #     q_nnode_bkup_gr_flag_fwd = True
                        # if (example_nnode_tps_rev[0] == _graph.types[0] and example_nnode_tps_rev[1] == _graph.types[1]) or \
                        #         (example_nnode_tps_rev[0] == _graph.types[1] and example_nnode_tps_rev[1] == _graph.types[0]):
                        #     q_nnode_bkup_gr_flag_rev = False
                        # else:
                        #     assert all(x in ['thing_1', 'thing_2'] for x in example_nnode_tps_rev)
                        #     q_nnode_bkup_gr_flag_rev = True
                    else:
                        raise AssertionError

                    if q_nearest_nodes is not None and len(q_nearest_nodes) == 0:
                        raise NotImplementedError
                except Exception as e:
                    print(e)
                    q_nnode_bkup_gr_flag = False
                    q_nnode_bkup_gr_flag_fwd = False
                    q_nnode_bkup_gr_flag_rev = False
                    q_nearest_nodes = [[[]], [[]]]
                    q_nearest_unodes_fwd = [[[]], [[]]]
                    q_nearest_unodes_rev = [[[]], [[]]]
            else:
                print(f"q_trped: Number of # is not equal to 2! ! {q_tpred_querytype} !")
                q_nnode_bkup_gr_flag = False
                q_nnode_bkup_gr_flag_fwd = False
                q_nnode_bkup_gr_flag_rev = False
                q_nearest_nodes = [[[]], [[]]]
                q_nearest_unodes_fwd = [[[]], [[]]]
                q_nearest_unodes_rev = [[[]], [[]]]
        elif smooth_h == 'wn':
            q_upredlist = upred2bow(q_upred, lang=lang)
            q_nearest_nodes = fetch_wn_smoothings(q_upredlist, direction='hyponym', gr_predstr2preds=_gr_predstr2preds,
                                                  first_only=True)
            if len(q_nearest_nodes[0][0]) > 0:
                q_nnode_bkup_gr_flag = False
                q_nearest_unodes_fwd = q_nearest_nodes
                q_nearest_unodes_rev = q_nearest_nodes
                q_nnode_bkup_gr_flag_fwd = False
                q_nnode_bkup_gr_flag_rev = False
            else:
                q_nearest_nodes = fetch_wn_smoothings(q_upredlist, direction='hyponym', gr_predstr2preds=_bkupgr_predstr2preds,
                                                      first_only=True)
                q_nearest_unodes_fwd = q_nearest_nodes
                q_nearest_unodes_rev = q_nearest_nodes
                q_nnode_bkup_gr_flag = True
                q_nnode_bkup_gr_flag_fwd = True
                q_nnode_bkup_gr_flag_rev = True

        else:
            raise AssertionError
    else:
        q_nearest_nodes = [[[]], [[]]]
        q_nearest_unodes_fwd = [[[]], [[]]]
        q_nearest_unodes_rev = [[[]], [[]]]
        q_nnode_bkup_gr_flag = False  # In this case, the flag will be checked, but no nearest nodes exist, so we just set it to False
        q_nnode_bkup_gr_flag_fwd = False
        q_nnode_bkup_gr_flag_rev = False

    num_true_entailments = 0.0
    num_samestr_true_entailments = 0.0
    num_smoothing = 0.0

    for ridx, rrel in enumerate(ref_rels):
        if num_prems_to_check is not None and ridx >= num_prems_to_check:
            break
        r_upred, r_subj, r_obj, r_tsubj, r_tobj = parse_rel(rrel)
        rup_halves = r_upred[r_upred.find('(') + 1:r_upred.find(')')].split(',')
        rup_left = rup_halves[0].split('.')
        if rup_left[-1].isdigit():
            rup_left = rup_left[:-1]
        else:
            print(f"``rup_left'' does not end with digits: {rup_left}!")
        if len(rup_left) == 0:
            return None, None, None, None, 0, 0, 0, False, False
        assert '_1' not in r_tsubj and '_2' not in r_tsubj and '_1' not in r_tobj and '_2' not in r_tobj
        if r_tsubj == r_tobj:
            # the assertion below seems deprecated, the ref argument types are allowed to be different from the query
            # argument types, but in these cases the ref argument types would be ignored.
            # assert q_tsubj[:-2] == q_tobj[:-2] and '_1' in q_tsubj and '_2' in q_tobj
            if rrel['aligned'] is True:
                r_tsubj = r_tsubj + '_1'
                r_tobj = r_tobj + '_2'
            elif rrel['aligned'] is False:
                r_tsubj = r_tsubj + '_2'
                r_tobj = r_tobj + '_1'
            else:
                raise AssertionError
        else:
            # assert q_tsubj != q_tobj and '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
            pass

        if rrel['aligned'] is True:
            r_tpred_querytype = '#'.join([r_upred, q_tsubj, q_tobj])
            r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
            r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
        elif rrel['aligned'] is False:
            r_tpred_querytype = '#'.join([r_upred, q_tobj, q_tsubj])
            r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
            r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
        else:
            raise AssertionError

        # p_smoothing (r_smoothing) checking
        if r_tpred_querytype in _graph.pred2Node:
            r_in_graph = True
            prm_all_missing_flag = False  # set the flag to False
        else:
            r_in_graph = False
            # if the current p is missing, we don't just set the prm_all_missing_flag to Ture:
            # we only do this when we are sure all premises are missing, that's at the end if the flag is kept as None.

        if smooth_p is not None:
            if smooth_p == 'lm':
                if len(r_tpred_querytype.split('#')) == 3:
                    assert len(r_tpred_graphtype_fwd.split('#')) == 3
                    assert len(r_tpred_graphtype_rev.split('#')) == 3
                    try:
                        if typematch_flag is True:
                            r_nearest_nodes = graph_deducer.get_nearest_node(preds=[r_tpred_querytype], k=smoothing_k,
                                                                             available_graphs=available_graph_types,
                                                                             target_typing=None,
                                                                             ablated_preds=[])
                            r_nearest_unodes_fwd = None
                            r_nearest_unodes_rev = None

                            example_nnode_tps = r_nearest_nodes[0][0][0].split('#')[1:]
                            # This is the reference, but the typing still follows the canonical q_types
                            if (example_nnode_tps[0] == q_tsubj and example_nnode_tps[1] == q_tobj) or \
                                    (example_nnode_tps[0] == q_tobj and example_nnode_tps[1] == q_tsubj):
                                r_nnode_bkup_gr_flag = False
                            else:
                                assert all(x in ['thing_1', 'thing_2'] for x in example_nnode_tps)
                                r_nnode_bkup_gr_flag = True
                            r_nnode_bkup_gr_flag_fwd = None
                            r_nnode_bkup_gr_flag_rev = None

                        elif typematch_flag is False:
                            r_nearest_nodes = None
                            r_nearest_unodes_fwd = None
                            r_nearest_unodes_rev = None
                            r_nnode_bkup_gr_flag = None
                            r_nnode_bkup_gr_flag_fwd = None
                            r_nnode_bkup_gr_flag_rev = None
                            # r_nearest_nodes = None
                            # r_nnode_bkup_gr_flag = None
                            # r_nearest_unodes_fwd = graph_deducer.get_nearest_node(preds=[r_tpred_graphtype_fwd], k=smoothing_k,
                            #                                                  available_graphs=available_graph_types,
                            #                                                  target_typing=None,
                            #                                                  ablated_preds=[])
                            # r_nearest_unodes_rev = graph_deducer.get_nearest_node(preds=[r_tpred_graphtype_rev], k=smoothing_k,
                            #                                                  available_graphs=available_graph_types,
                            #                                                  target_typing=None,
                            #                                                  ablated_preds=[])
                            #
                            # example_nnode_tps_fwd = r_nearest_unodes_fwd[0][0][0].split('#')[1:]
                            # example_nnode_tps_rev = r_nearest_unodes_rev[0][0][0].split('#')[1:]
                            #
                            # if (example_nnode_tps_fwd[0] == _graph.types[0] and example_nnode_tps_fwd[1] == _graph.types[1]) or \
                            #         (example_nnode_tps_fwd[0] == _graph.types[1] and example_nnode_tps_fwd[1] == _graph.types[0]):
                            #     r_nnode_bkup_gr_flag_fwd = False
                            # else:
                            #     assert all(x in ['thing_1', 'thing_2'] for x in example_nnode_tps_fwd)
                            #     r_nnode_bkup_gr_flag_fwd = True
                            # if (example_nnode_tps_rev[0] == _graph.types[0] and example_nnode_tps_rev[1] == _graph.types[1]) or \
                            #         (example_nnode_tps_rev[0] == _graph.types[1] and example_nnode_tps_rev[1] == _graph.types[0]):
                            #     r_nnode_bkup_gr_flag_rev = False
                            # else:
                            #     assert all(x in ['thing_1', 'thing_2'] for x in example_nnode_tps_rev)
                            #     r_nnode_bkup_gr_flag_rev = True

                        else:
                            raise AssertionError
                    except Exception as e:
                        print(e)
                        r_nnode_bkup_gr_flag = False
                        r_nnode_bkup_gr_flag_fwd = False
                        r_nnode_bkup_gr_flag_rev = False
                        r_nearest_nodes = [[[]], [[]]]
                        r_nearest_unodes_fwd = [[[]], [[]]]
                        r_nearest_unodes_rev = [[[]], [[]]]
                else:
                    print(f"r_trped: Number of # is not equal to 2! ! {r_tpred_querytype} !")
                    r_nnode_bkup_gr_flag = False
                    r_nnode_bkup_gr_flag_fwd = False
                    r_nnode_bkup_gr_flag_rev = False
                    r_nearest_nodes = [[[]], [[]]]
                    r_nearest_unodes_fwd = [[[]], [[]]]
                    r_nearest_unodes_rev = [[[]], [[]]]
            elif smooth_p == 'wn':
                r_upredlist = upred2bow(r_upred, lang=lang)
                r_nearest_nodes = fetch_wn_smoothings(r_upredlist, direction='hypernym', gr_predstr2preds=_gr_predstr2preds,
                                                      first_only=True)
                if len(r_nearest_nodes[0][0]) > 0:
                    r_nearest_unodes_fwd = r_nearest_nodes
                    r_nearest_unodes_rev = r_nearest_nodes
                    r_nnode_bkup_gr_flag = False
                    r_nnode_bkup_gr_flag_fwd = False
                    r_nnode_bkup_gr_flag_rev = False
                else:
                    r_nearest_nodes = fetch_wn_smoothings(r_upredlist, direction='hypernym', gr_predstr2preds=_bkupgr_predstr2preds,
                                                          first_only=True)
                    r_nearest_unodes_fwd = r_nearest_nodes
                    r_nearest_unodes_rev = r_nearest_nodes
                    r_nnode_bkup_gr_flag = True
                    r_nnode_bkup_gr_flag_fwd = True
                    r_nnode_bkup_gr_flag_rev = True
            else:
                raise AssertionError
        else:
            r_nearest_nodes = [[[]], [[]]]
            r_nearest_unodes_fwd = [[[]], [[]]]
            r_nearest_unodes_rev = [[[]], [[]]]
            r_nnode_bkup_gr_flag = False  # In this case, the flag will be checked, but no nearest nodes exist, so we just set it to False
            r_nnode_bkup_gr_flag_fwd = False
            r_nnode_bkup_gr_flag_rev = False

        # print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
        # TODO: Attention! Check this feat_idx matter!
        effective_feat_idx = feat_idx + 0
        if typematch_flag is True:
            assert (q_tsubj == _graph.types[0] and q_tobj == _graph.types[1]) or \
                   (q_tsubj == _graph.types[1] and q_tobj == _graph.types[0])

            if q_in_graph is True and r_in_graph is True:

                curr_res_tscore, curr_res_tscore_ref, is_direct_true_ent_flag, is_samestr_true_ent_flag \
                    = get_fuzzy_score_for_pair(_graph, r_tpred_querytype, q_tpred_querytype, all_predstr_to_upreds,
                                               effective_feat_idx, lang, threshold_samestr, debug)
                if maximum_tscore is None or curr_res_tscore > maximum_tscore:
                    maximum_tscore = curr_res_tscore
                    max_tscore_ref = curr_res_tscore_ref
                if is_direct_true_ent_flag:
                    num_true_entailments += 1
                elif is_samestr_true_ent_flag:
                    num_samestr_true_entailments += 1

            if q_in_graph is True:
                # This function is for each query, outer loop is for each reference, this inner loop below is for
                # each neighbor-node.
                if r_nnode_bkup_gr_flag is True:  # then, convert the q_node to thing#thing backup type as well, and use the backup_graph for inference.
                    q_tpred_smoothtype = get_backuptype_tpred(q_tpred_querytype, rrel['aligned'])
                    smooth_gr = _backup_graph
                elif r_nnode_bkup_gr_flag is False:
                    q_tpred_smoothtype = q_tpred_querytype
                    smooth_gr = _graph
                else:
                    raise AssertionError
                for curr_refe_subs_tpred, curr_refe_subs_scr in zip(r_nearest_nodes[0][0], r_nearest_nodes[1][0]):
                    curr_res_tsscore, curr_res_tsscore_ref, _, _ = get_fuzzy_score_for_pair(smooth_gr, curr_refe_subs_tpred,
                                                                                    q_tpred_smoothtype, all_predstr_to_upreds,
                                                                                    effective_feat_idx, lang,
                                                                                    threshold_samestr, debug)
                    if curr_res_tsscore is not None:
                        curr_refe_subs_scr = curr_refe_subs_scr ** smooth_sim_order  # bring this similarity value closer to 1
                        curr_res_tsscore *= curr_refe_subs_scr
                        if maximum_tsscore is None or curr_res_tsscore > maximum_tsscore:
                            maximum_tsscore = curr_res_tsscore
                            max_tsscore_ref = curr_res_tsscore_ref

            if r_in_graph is True:
                if q_nnode_bkup_gr_flag is True:
                    r_tpred_smoothtype = get_backuptype_tpred(r_tpred_querytype, True)
                    smooth_gr = _backup_graph
                elif q_nnode_bkup_gr_flag is False:
                    r_tpred_smoothtype = r_tpred_querytype
                    smooth_gr = _graph
                else:
                    raise AssertionError
                for curr_query_subs_tpred, curr_query_subs_scr in zip(q_nearest_nodes[0][0], q_nearest_nodes[1][0]):

                    curr_res_tsscore, curr_res_tsscore_ref, _, _ = get_fuzzy_score_for_pair(smooth_gr, r_tpred_smoothtype,
                                                                                    curr_query_subs_tpred, all_predstr_to_upreds,
                                                                                    effective_feat_idx, lang,
                                                                                    threshold_samestr, debug)
                    if curr_res_tsscore is not None:
                        curr_query_subs_scr = curr_query_subs_scr ** smooth_sim_order  # bring this similarity value closer to 1
                        curr_res_tsscore *= curr_query_subs_scr
                        if maximum_tsscore is None or curr_res_tsscore > maximum_tsscore:
                            maximum_tsscore = curr_res_tsscore
                            max_tsscore_ref = curr_res_tsscore_ref

            # below: smoothing both q and r
            if q_nnode_bkup_gr_flag is True and r_nnode_bkup_gr_flag is True:
                # if both are backup types, then both don't need further conversion
                r_nearest_nodes_smoothtype = r_nearest_nodes
                q_nearest_nodes_smoothtype = q_nearest_nodes
                smooth_gr = _backup_graph
            elif q_nnode_bkup_gr_flag is True and r_nnode_bkup_gr_flag is False:
                # if q is backup types and r is not, then r needs conversion
                q_nearest_nodes_smoothtype = q_nearest_nodes
                r_nearest_nodes_smoothtype = [[[]], [[]]]
                for refe_subs_tpred, refe_subs_scr in zip(r_nearest_nodes[0][0], r_nearest_nodes[1][0]):
                    refe_subs_tpred_smoothtype = get_backuptype_tpred(refe_subs_tpred, True)
                    r_nearest_nodes_smoothtype[0][0].append(refe_subs_tpred_smoothtype)
                    r_nearest_nodes_smoothtype[1][0].append(refe_subs_scr)
                smooth_gr = _backup_graph
            elif q_nnode_bkup_gr_flag is False and r_nnode_bkup_gr_flag is True:
                # if r is backup types and q is not, then q needs conversion, ``aligned'' may be false
                r_nearest_nodes_smoothtype = r_nearest_nodes
                q_nearest_nodes_smoothtype = [[[]], [[]]]
                for query_subs_tpred, query_subs_scr in zip(q_nearest_nodes[0][0], q_nearest_nodes[1][0]):
                    query_subs_tpred_smoothtype = get_backuptype_tpred(query_subs_tpred, rrel['aligned'])
                    q_nearest_nodes_smoothtype[0][0].append(query_subs_tpred_smoothtype)
                    q_nearest_nodes_smoothtype[1][0].append(query_subs_scr)
                smooth_gr = _backup_graph
            elif q_nnode_bkup_gr_flag is False and r_nnode_bkup_gr_flag is False:
                r_nearest_nodes_smoothtype = r_nearest_nodes
                q_nearest_nodes_smoothtype = q_nearest_nodes
                smooth_gr = _graph
            else:
                raise AssertionError

            try:
                a = r_nearest_nodes_smoothtype[0][0]
                a = r_nearest_nodes_smoothtype[1][0]
                a = q_nearest_nodes_smoothtype[0][0]
                a = q_nearest_nodes_smoothtype[1][0]
            except Exception as e:
                print(f"r_nearest_nodes_smoothtype: {r_nearest_nodes_smoothtype}")
                print(f"q_nearest_nodes_smoothtype: {q_nearest_nodes_smoothtype}")

            for curr_refe_subs_tpred, curr_refe_subs_scr in zip(r_nearest_nodes_smoothtype[0][0], r_nearest_nodes_smoothtype[1][0]):
                for curr_query_subs_tpred, curr_query_subs_scr in zip(q_nearest_nodes_smoothtype[0][0], q_nearest_nodes_smoothtype[1][0]):
                    curr_res_tsscore, curr_res_tsscore_ref, _, _ = get_fuzzy_score_for_pair(smooth_gr, curr_refe_subs_tpred,
                                                                                curr_query_subs_tpred, all_predstr_to_upreds,
                                                                                effective_feat_idx, lang,
                                                                                threshold_samestr, debug)
                    if curr_res_tsscore is not None:
                        curr_refe_subs_scr = curr_refe_subs_scr ** smooth_sim_order
                        curr_query_subs_scr = curr_query_subs_scr ** smooth_sim_order
                        curr_res_tsscore *= (curr_refe_subs_scr * curr_query_subs_scr)
                        if maximum_tsscore is None or curr_res_tsscore > maximum_tsscore:
                            maximum_tsscore = curr_res_tsscore
                            max_tsscore_ref = curr_res_tsscore_ref
            num_smoothing += 1

        else:
            num_true_entailments = None
            num_samestr_true_entailments = None
            num_smoothing = None

        cur_uscores_fwd = _graph.get_features(r_tpred_graphtype_fwd, q_tpred_graphtype_fwd)
        cur_uscores_rev = _graph.get_features(r_tpred_graphtype_rev, q_tpred_graphtype_rev)

        if cur_uscores_fwd is not None:
            if cur_uscores_fwd[1] > 0 and cur_uscores_fwd[1] < 0.99:
                # print(cur_uscores_fwd)
                # [0.43617837 0.2739726  0.16573886 0.18112025 0.13438165 0.09970408, 0.5        0.5        1.         0.5        1.         1.        ]
                # print("!")
                pass
            curfeat_cur_uscore_fwd = cur_uscores_fwd[effective_feat_idx]
            if maximum_uscore is None or curfeat_cur_uscore_fwd > maximum_uscore:
                max_uscore_ref = r_tpred_graphtype_fwd
                maximum_uscore = curfeat_cur_uscore_fwd
        else:
            curfeat_cur_uscore_fwd = None
        if cur_uscores_rev is not None:
            curfeat_cur_uscore_rev = cur_uscores_rev[effective_feat_idx]
            if maximum_uscore is None or curfeat_cur_uscore_rev > maximum_uscore:
                max_uscore_ref = r_tpred_graphtype_rev
                maximum_uscore = curfeat_cur_uscore_rev
        else:
            curfeat_cur_uscore_rev = None

        # the last condition means that we do back off to other upreds with the same pred_str
        if (curfeat_cur_uscore_fwd is None or curfeat_cur_uscore_fwd == 0) and all_predstr_to_upreds is not None:
            curfeat_samestr_uscore_fwd, maxi_ref_tpred = find_samestr_scores(r_tpred_graphtype_fwd, q_tpred_graphtype_fwd,
                                                                             all_predstr_to_upreds, _graph,
                                                                             effective_feat_idx, reduce='max', lang=lang,
                                                                             threshold_samestr=threshold_samestr,
                                                                             debug=debug)
            if curfeat_samestr_uscore_fwd is not None:
                assert maxi_ref_tpred is not None
                if maximum_uscore is None or curfeat_samestr_uscore_fwd > maximum_uscore:
                    max_uscore_ref = maxi_ref_tpred
                    maximum_uscore = curfeat_samestr_uscore_fwd

        # the last condition means that we do back off to other upreds with the same pred_str
        if (curfeat_cur_uscore_rev is None or curfeat_cur_uscore_rev == 0) and all_predstr_to_upreds is not None:
            curfeat_samestr_uscore_rev, maxi_ref_tpred = find_samestr_scores(r_tpred_graphtype_rev,
                                                                             q_tpred_graphtype_rev,
                                                                             all_predstr_to_upreds, _graph,
                                                                             effective_feat_idx, reduce='max', lang=lang,
                                                                             threshold_samestr=threshold_samestr,
                                                                             debug=debug)
            if curfeat_samestr_uscore_rev is not None:
                assert maxi_ref_tpred is not None
                if maximum_uscore is None or curfeat_samestr_uscore_rev > maximum_uscore:
                    max_uscore_ref = maxi_ref_tpred
                    maximum_uscore = curfeat_samestr_uscore_rev

        # TODO: get untyped smoothing scores.
        # maximum_usscore = XXX
        # max_usscore_ref = XXX

    if prm_all_missing_flag is None:  # if the prm_all_missing_flag has not been set to False, that means it is True.
        prm_all_missing_flag = True
    else:
        assert prm_all_missing_flag is False

    if debug and maximum_tscore is not None:
        print(f"query: {q_tpred_querytype}; max_tscore_ref: {max_tscore_ref}; max_uscore_ref: {max_uscore_ref};"
              f"max_tsscore_ref: {max_tsscore_ref};")

    # At the end of the function, we assert that both prm_all_missing_flag and hyp_missing_flag has been set (are not None);
    assert prm_all_missing_flag is not None
    assert hyp_missing_flag is not None

    return maximum_tscore, maximum_uscore, maximum_tsscore, maximum_usscore, num_true_entailments, \
           num_samestr_true_entailments, num_smoothing, prm_all_missing_flag, hyp_missing_flag


def find_answers_from_graph(_graph, ent, ref_rels, typematch_flag, partial_typematch_flag, feat_idx, debug=False):
    # find entailment matches for one entry from one graph
    this_ent_rtscores_bucket = {} if typematch_flag else None
    this_ent_ftscores_bucket = {} if partial_typematch_flag else None
    this_ent_uscores_bucket = {}

    q_upred, q_subj, q_obj, q_tsubj, q_tobj = parse_rel(ent)
    assert '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
    if q_tsubj == q_tobj:
        q_tsubj = q_tsubj + '_1'
        q_tobj = q_tobj + '_2'
    q_tpred_querytype = '#'.join([q_upred, q_tsubj, q_tobj])
    assert len(_graph.types) == 2
    q_tpred_graphtype_fwd = '#'.join([q_upred, _graph.types[0], _graph.types[1]])
    q_tpred_graphtype_rev = '#'.join([q_upred, _graph.types[1], _graph.types[0]])

    num_true_entailments = 0.0

    for rrel in ref_rels:
        r_upred, r_subj, r_obj, r_tsubj, r_tobj = parse_rel(rrel)

        if r_subj == ent['index_arg']:
            r_ans = r_obj
        elif r_obj == ent['index_arg']:
            r_ans = r_subj
        else:
            raise AssertionError

        assert '_1' not in r_tsubj and '_2' not in r_tsubj and '_1' not in r_tobj and '_2' not in r_tobj
        if r_tsubj == r_tobj:
            # the assertion below seems deprecated, the ref argument types are allowed to be different from the query
            # argument types, but in these cases the ref argument types would be ignored.
            # assert q_tsubj[:-2] == q_tobj[:-2] and '_1' in q_tsubj and '_2' in q_tobj
            if rrel['aligned'] is True:
                r_tsubj = r_tsubj + '_1'
                r_tobj = r_tobj + '_2'
            elif rrel['aligned'] is False:
                r_tsubj = r_tsubj + '_2'
                r_tobj = r_tobj + '_1'
            else:
                raise AssertionError
        else:
            # assert q_tsubj != q_tobj and '_1' not in q_tsubj and '_2' not in q_tsubj and '_1' not in q_tobj and '_2' not in q_tobj
            pass

        if rrel['aligned'] is True:
            r_tpred_querytype = '#'.join([r_upred, q_tsubj, q_tobj])
            r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
            r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
        elif rrel['aligned'] is False:
            r_tpred_querytype = '#'.join([r_upred, q_tobj, q_tsubj])
            r_tpred_graphtype_fwd = '#'.join([r_upred, _graph.types[1], _graph.types[0]])
            r_tpred_graphtype_rev = '#'.join([r_upred, _graph.types[0], _graph.types[1]])
        else:
            raise AssertionError

        # print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
        # TODO: Attention! Check this feat_idx matter!
        effective_feat_idx = feat_idx + 0
        if typematch_flag is True:
            assert (q_tsubj == _graph.types[0] and q_tobj == _graph.types[1]) or \
                   (q_tsubj == _graph.types[1] and q_tobj == _graph.types[0])

            cur_tscores = _graph.get_features(r_tpred_querytype, q_tpred_querytype)
            if cur_tscores is not None:
                # print(f"cur tscores length: {len(cur_tscores)}")
                # print(f"Attention! Check this feat_idx matter!", file=sys.stderr)
                if cur_tscores[0] > 0 and cur_tscores[0] < 0.99:
                    num_true_entailments += 1
                curfeat_cur_tscore = cur_tscores[effective_feat_idx]

                if r_ans not in this_ent_rtscores_bucket:
                    this_ent_rtscores_bucket[r_ans] = 0.0

                if curfeat_cur_tscore > this_ent_rtscores_bucket[r_ans]:
                    this_ent_rtscores_bucket[r_ans] = curfeat_cur_tscore
        else:
            num_true_entailments = None

        cur_gt_scores_fwd = _graph.get_features(r_tpred_graphtype_fwd, q_tpred_graphtype_fwd)
        cur_gt_scores_rev = _graph.get_features(r_tpred_graphtype_rev, q_tpred_graphtype_rev)

        if partial_typematch_flag is True:
            # these additional assertions are for the cases where, for instance, the query type has only one `art' but the
            # graph type is art_1#art_2
            assert ent['index_type'] == _graph.types[0] or ent['index_type'] == _graph.types[1] or \
                   (ent['index_type'] == _graph.types[0][:-2] and _graph.types[0][-2:] in ['_1', '_2']) or \
                   (ent['index_type'] == _graph.types[1][:-2] and _graph.types[1][-2:] in ['_1', '_2'])
            # only take the score with index type in the correct position!
            if ent['index_type'] in [_graph.types[0], _graph.types[0][:-2]] and ent['index_position'] == 'subj':
                gtype_order_for_query = 'fwd'
            elif ent['index_type'] in [_graph.types[1], _graph.types[1][:-2]] and ent['index_position'] == 'obj':
                gtype_order_for_query = 'fwd'
            elif ent['index_type'] in [_graph.types[0], _graph.types[0][:-2]] and ent['index_position'] == 'obj':
                gtype_order_for_query = 'rev'
            elif ent['index_type'] in [_graph.types[1], _graph.types[0][:-2]] and ent['index_position'] == 'subj':
                gtype_order_for_query = 'rev'
            else:
                raise AssertionError

            if gtype_order_for_query == 'fwd':
                cur_tscores = cur_gt_scores_fwd
            elif gtype_order_for_query == 'rev':
                cur_tscores = cur_gt_scores_rev
            else:
                raise AssertionError

            if cur_tscores is not None:
                curfeat_cur_tscore = cur_tscores[effective_feat_idx]
                if r_ans not in this_ent_ftscores_bucket:
                    this_ent_ftscores_bucket[r_ans] = 0.0
                if curfeat_cur_tscore > this_ent_ftscores_bucket[r_ans]:
                    this_ent_ftscores_bucket[r_ans] = curfeat_cur_tscore
        else:
            pass

        cur_uscores_fwd = cur_gt_scores_fwd
        cur_uscores_rev = cur_gt_scores_rev

        if cur_uscores_fwd is not None:
            if cur_uscores_fwd[1] > 0 and cur_uscores_fwd[1] < 0.99 and debug:
                # print(cur_uscores_fwd)
                # [0.43617837 0.2739726  0.16573886 0.18112025 0.13438165 0.09970408, 0.5        0.5        1.         0.5        1.         1.        ]
                # print("!")
                pass
            curfeat_cur_uscore_fwd = cur_uscores_fwd[effective_feat_idx]
            if r_ans not in this_ent_uscores_bucket:
                this_ent_uscores_bucket[r_ans] = 0.0
            if curfeat_cur_uscore_fwd > this_ent_uscores_bucket[r_ans]:
                this_ent_uscores_bucket[r_ans] = curfeat_cur_uscore_fwd

        if cur_uscores_rev is not None:
            curfeat_cur_uscore_rev = cur_uscores_rev[effective_feat_idx]
            if r_ans not in this_ent_uscores_bucket:
                this_ent_uscores_bucket[r_ans] = 0.0
            if curfeat_cur_uscore_rev > this_ent_uscores_bucket[r_ans]:
                this_ent_uscores_bucket[r_ans] = curfeat_cur_uscore_rev

    if debug and this_ent_rtscores_bucket is not None:
        print(f"query: {q_tpred_querytype}; answers bucket: ")
        for ans in this_ent_rtscores_bucket:
            print(f"{ans}: {this_ent_rtscores_bucket[ans]};")

    this_ent_rtscores_bucket = None if this_ent_rtscores_bucket is None else {a: s for (a, s) in sorted(this_ent_rtscores_bucket.items(), key=lambda x: x[1], reverse=True)[:50]}
    this_ent_ftscores_bucket = None if this_ent_ftscores_bucket is None else {a: s for (a, s) in sorted(this_ent_ftscores_bucket.items(), key=lambda x: x[1], reverse=True)[:50]}
    this_ent_uscores_bucket = None if this_ent_uscores_bucket is None else {a: s for (a, s) in sorted(this_ent_uscores_bucket.items(), key=lambda x: x[1], reverse=True)[:50]}

    return this_ent_rtscores_bucket, this_ent_ftscores_bucket, this_ent_uscores_bucket, num_true_entailments
