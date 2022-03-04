from collections import OrderedDict
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk import pos_tag, RegexpParser, Tree
import numpy as np
import torch
from utils_nltk import parse_ner_results

from copy import copy

NP = "NP: {(<V\w+>|<NN\w?>|<JJ\w?>)+.*<NN\w?>}"
chunker = RegexpParser(NP)


def get_continuous_chunks(text, chunk_func):
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk


def create_mapping(sentence, return_pt=False, nlp = None, tokenizer=None, ner=None):
    '''
    Create a mapping
    tokenizer: huggingface tokenizer
    '''
    tokenizer_mwe = MWETokenizer(separator=" ")
    
    noun_chunks = get_continuous_chunks(sentence, chunk_func=chunker.parse)
    for chunk in noun_chunks:
        chunk_tokens = word_tokenize(chunk)
        tokenizer_mwe.add_mwe(tuple(chunk_tokens))
        
    for parsed_candidate in ner:
        tokenizer_mwe.add_mwe(tuple(word_tokenize(parsed_candidate)))

    single_tokens = word_tokenize(sentence)
    sentence_mapping = tokenizer_mwe.tokenize(single_tokens)
 
    token2id = {}
    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        if token not in token2id:
            token2id[token] = len(token2id)

        subtoken_ids = tokenizer(str(token), add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [ token2id[token] ]*len(subtoken_ids)
        token_ids += subtoken_ids

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)
    
    return outputs, tokenid2word_mapping, token2id, sentence_mapping, noun_chunks


def create_mapping_target(sentence, target, return_pt=False, tokenizer=None):
    tokenizer_mwe = MWETokenizer(separator=" ")
    single_tokens = word_tokenize(sentence)

    for triplet in target:
        for phrase in triplet[:-1]:
            phrase_tokens = word_tokenize(phrase)
            if len(phrase_tokens) > 1:
                tokenizer_mwe.add_mwe(tuple(phrase_tokens))

    sentence_mapping = tokenizer_mwe.tokenize(single_tokens)

    token2id = {}
    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        if token not in token2id:
            token2id[token] = len(token2id)

        subtoken_ids = tokenizer(str(token), add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [ token2id[token] ]*len(subtoken_ids)
        token_ids += subtoken_ids

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)

    return outputs, tokenid2word_mapping, token2id, sentence_mapping


def compress_attention(attention, tokenid2word_mapping, operator=np.mean):

    new_index = []
    
    prev = -1
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    attention = np.array(new_matrix).T

    prev = -1
    new_index = []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append([row])
            prev = token_id
        else:
            new_index[-1].append(row)

    
    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))
    
    new_matrix = np.array(new_matrix)

    return new_matrix.T


def index2word(tokenid2word_mapping, token2id):
    tokens = []
    prev = -1
    for token_id in tokenid2word_mapping:
        if token_id == prev:
            continue

        tokens.append(token2id[token_id])
        prev = token_id

    return tokens


def parse_ner_results(ner_results):
    candidates = []
    sub_fold = []
    for idx, curr in enumerate(ner_results):
        if idx == 0:
            sub_fold.append(curr['word'])
            prev_flag = curr['entity'].split('-')[0]
            prev = curr
            continue
        curr_flag = curr['entity'].split('-')[0]
        if prev_flag == 'B' and curr_flag == 'B' and not idx:
            candidates.append(sub_fold[0])
            sub_fold = []

        elif prev_flag == 'B' and curr_flag == 'B' and idx:
            sub_fold.append(prev['word'])
            candidates.append(sub_fold[0])
            sub_fold = []
            sub_fold.append(curr['word'])

        elif prev_flag == 'B' and curr_flag == 'I':
            sub_fold.append(prev['word'])
            sub_fold.append(curr['word'])

        elif (prev_flag == 'I') and (curr_flag == 'I' ) and (idx + 1 < len(ner_results)):
            sub_fold.append(curr['word'])

        elif (prev_flag == 'I') and (curr_flag == 'B' ):
            ordered = OrderedDict(dict(zip(sub_fold, range(len(sub_fold)))))
            candidates.append(' '.join(list(ordered.keys())).replace(' #', '').replace('#', ''))
            sub_fold = []
            sub_fold.append(curr['word'])

        elif (prev_flag == 'I') and (curr_flag == 'I' ) and (idx + 1) == len(ner_results):
            sub_fold.append(curr['word'])
            ordered = OrderedDict(dict(zip(sub_fold, range(len(sub_fold)))))
            candidates.append(' '.join(list(ordered.keys())))
            sub_fold = []

        prev = curr
        prev_flag = prev['entity'].split('-')[0]
    return candidates

