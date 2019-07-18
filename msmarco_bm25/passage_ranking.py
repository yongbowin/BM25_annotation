"""
Author: Yongbo Wang
Date: 7/17/2019 15:25 PM
Description: For passages ranking in MS MARCO.
"""
import argparse
import json
import operator
import itertools
from nltk import sent_tokenize, word_tokenize
import string
from math import log
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_text(text):
    """
    Text tokenize: Sentence tokenize --> Word tokenize --> Replace specific char
    """
    tokens_list = list(
        itertools.chain.from_iterable(
            (token for token in word_tokenize(sent))
            for sent in sent_tokenize(text)))

    return tokens_list


def remove_punc(tokens_list):
    """
    Remove en punc.
    """
    tok_no_punc = []
    for token in tokens_list:
        if token not in string.punctuation:
            tok_no_punc.append(token)

    return tok_no_punc


def read_data(input_file, quotechar=None):
    """
    Read query and passages text data.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    idx_tok_dict = {}
    for line in tqdm(lines):
        sample = json.loads(line)
        query_id = sample["query_id"]
        query_tok = tokenize_text(sample["query"])
        passage_text_tok_list = []
        for idx, doc in enumerate(sample["passages"]):
            passage_text_tok = tokenize_text(doc["passage_text"])  # Tokenize
            passage_text_tok_list.append(passage_text_tok)  # Tokenize

        # Remove punc
        query_tok_no_punc = remove_punc(query_tok)
        passage_tok_no_punc_list = []
        for item_list in passage_text_tok_list:
            passage_tok_no_punc_list.append(remove_punc(item_list))

        idx_tok_dict[query_id] = {
            "query_tok": query_tok_no_punc,
            "passages_tok": passage_tok_no_punc_list
        }

    return idx_tok_dict


def score_BM25(n, f, qf, r, N, dl, avdl):
    """
    Compute BM25 score.
    """
    k1 = 1.2
    k2 = 100
    b = 0.75
    R = 0.0

    K = k1 * ((1-b) + b * (float(dl)/float(avdl)))
    first = log(
        ((r + 0.5)/(R - r + 0.5)) / ((n - r + 0.5)/(N - n - R + r + 0.5))
    )
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2+1) * qf) / (k2 + qf)

    return first * second * third


def build_data_structures(corpus):
    # idx = InvertedIndex()
    # dlt = DocumentLengthTable()
    index = dict()
    table = dict()
    for docid in corpus:
        # build inverted index
        for word in corpus[docid]:
            # idx_add(str(word), str(docid))
            if str(word) in index:
                if str(docid) in index[str(word)]:
                    index[str(word)][str(docid)] += 1
                else:
                    index[str(word)][str(docid)] = 1
            else:
                d = dict()
                d[str(docid)] = 1
                index[str(word)] = d

        # build document length table
        length = len(corpus[str(docid)])
        table[docid] = length  # {"id": length, ...}
    return index, table


def passage_ranking(idx_tok_dict):
    """
    Ranking passages.
    """
    ranking_res = {}
    for k, v in tqdm(idx_tok_dict.items()):  # k is `query_id`, for one sample
        corpus = {}
        for idx, text_tok in enumerate(v["passages_tok"]):
            corpus[str(idx)] = text_tok
        index, table = build_data_structures(corpus)  # corpus is {"num": tok_list, ...}

        # avg length
        sum = 0
        for length in table.values():
            sum += length
        avg_length = float(sum) / float(len(table))

        query_result = dict()
        for term in v["query_tok"]:  # for each word in query
            if term in index:  # doc words in `self.index`, self.index[word][docid]
                doc_dict = index[term]  # retrieve index entry
                """
                for each document and its (the current query word's) word frequency,
                """
                for docid, freq in doc_dict.items():
                    docid = str(docid)
                    score = score_BM25(
                        n=len(doc_dict),
                        f=freq,
                        qf=1,
                        r=0,
                        N=len(table),
                        dl=table[docid],  # length
                        avdl=avg_length)  # calculate score
                    if docid in query_result:  # this document has already been scored once
                        query_result[docid] += score
                    else:
                        query_result[docid] = score

        # sort by score
        sorted_x = sorted(query_result.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        index = 0
        ranking_res[k] = []
        for i in sorted_x:
            # query_id, doc_num, index, score
            tmp = (k, i[0], index, i[1])
            ranking_res[k].append([index, i[0], i[1]])  # index, doc_num, score
            # `{:>1}` denotes 右对齐 (宽度为1)
            # print('{:>1}\t{:>4}\t{:>2}\t{:>12}\tBM25'.format(*tmp))
            index += 1

    return ranking_res


if __name__ == '__main__':
    """  
    export BASE_PATH=/DATA2/disk1/wangyongbo/ms_marco/data
    
    python passage_ranking.py --input_file $BASE_PATH/lines_train_v2.1.json --output_file $BASE_PATH/lines_train_v2.1_ranked.json
    """
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True, help="The file will be ranked.")
    parser.add_argument("--output_file", default=None, type=str, required=True, help="The ranked output.")
    args = parser.parse_args()

    idx_tok_dict = read_data(args.input_file)  # read data
    ranking_res = passage_ranking(idx_tok_dict)  # ranking passages

    with open(args.output_file, 'w') as fout:
        fout.write(json.dumps(ranking_res, ensure_ascii=False))
