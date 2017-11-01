# -*- coding: utf-8 -*-
import gensim
import os
import collections
import MeCab
import time
import sys
import pickle
from langdetect import detect
import re
from doc2vec_utils import BuildSentence
import numpy as np
from infer_vector import GetDoc2vecVector
import matplotlib.pyplot as plt
import nltk

data_dir = '/home/satoshi/dev/kaggle/spooky/data/text'
min_word_count = 0
phrase_file = ''
train_corpus_pickle = ''

# mt = MeCab.Tagger("-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
mt = MeCab.Tagger("-d /usr/lib/mecab/dic/mecab-ipadic-neologd")
asciiReg = re.compile(r'^[!-~]+$')
hira_regexp = re.compile(r'[\u3040-\u309f]')
kata_regexp = re.compile(r'[\u30a0-\u30ff]')
phrase_regexp = re.compile(r'^.+_.+$')
alpha_regexp = re.compile(r'[^a-zA-Z_]')

vector_size = 300
min_count = 5
iter = 15
workers = 10
window_size = 15

get_doc2vec_vector = GetDoc2vecVector('spooky.model')

def read_corpus_with_phrase(max_data_count=10000):
    train_corpus = []
    if os.path.exists(train_corpus_pickle):
        with open(train_corpus_pickle, mode='rb') as f:
            train_corpus = pickle.load(f)
        return train_corpus

    bs = BuildSentence(mt)

    corpus_cnt = 0
    titles = []
    sentences = []
    for filename in [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]:
        if corpus_cnt >= max_data_count:
            break

        try:
            with open(os.path.join(data_dir, filename)) as f:
                title = filename.replace(".txt", "")
                content = f.read()
        except:
            pass
        #content = gensim.utils.simple_preprocess(content, min_len=1)
        content = bs.nlp_processing(content)
        titles.append(title)
        sentences.append(content + content)
        corpus_cnt += 1

    # sentences_1 = []
    # for i, s in enumerate(m[sentences]):
    #    new_s = build_sentence(s, stop_words=[])
    #    sentences_1.append(new_s)

    # m.add_vocab(sentences_1)
    # for i, s in enumerate(m[sentences_1]):
    '''
    sentences_work = np.array(sentences)
    count_list = np.array([len(s) for s in sentences_work])
    print(count_list.max())
    print(count_list.min())
    plt.hist(count_list,bins=10)
    plt.show()
    '''
    for i, s in enumerate(sentences):
        # new_s = bs.build_sentence(s)
        tagged_doc = gensim.models.doc2vec.TaggedDocument(s, [titles[i]])
        if len(tagged_doc.words) < min_word_count:
            continue
        train_corpus.append(tagged_doc)

    return train_corpus


def generate_class_frequent_word_list(dictionary, rank_vector_dic_list):
    a_tfidf_list = []
    b_tfidf_list = []
    c_tfidf_list = []
    s_tfidf_list = []
    for rank_vector in rank_vector_dic_list:
        rank = rank_vector['r']
        if rank == 'A':
            a_tfidf_list.append(rank_vector['tv'])
        elif rank == 'B':
            b_tfidf_list.append(rank_vector['tv'])
        elif rank == 'C':
            c_tfidf_list.append(rank_vector['tv'])
        else:
            s_tfidf_list.append(rank_vector['tv'])
    a_sum_list = np.sum(a_tfidf_list, axis=0)
    a_sort_index_list = a_sum_list.argsort()[::-1]
    a_top_word_list = [dictionary[id] for id in a_sort_index_list]
    b_sum_list = np.sum(b_tfidf_list, axis=0)
    b_sort_index_list = b_sum_list.argsort()[::-1]
    b_top_word_list = [dictionary[id] for id in b_sort_index_list]
    c_sum_list = np.sum(c_tfidf_list, axis=0)
    c_sort_index_list = c_sum_list.argsort()[::-1]
    c_top_word_list = [dictionary[id] for id in c_sort_index_list]
    s_sum_list = np.sum(s_tfidf_list, axis=0)
    s_sort_index_list = s_sum_list.argsort()[::-1]
    s_top_word_list = [dictionary[id] for id in s_sort_index_list]

    with open('top_word_list.csv', 'w') as f:
        f.write('S,'+','.join(s_top_word_list[0:100]) + '\n')
        f.write('A,'+','.join(a_top_word_list[0:100]) + '\n')
        f.write('B,'+','.join(b_top_word_list[0:100]) + '\n')
        f.write('C,'+','.join(c_top_word_list[0:100]) + '\n')

    return


def generat_vector(train_corpus):
    st = time.time()

    # 辞書の作成, 単語のIDを生成する
    doc_word_list = []
    for td in train_corpus:
        doc_word_list.append(td.words)
    dictionary = gensim.corpora.Dictionary(doc_word_list)
    # 文章の単語の出現回数のカウント
    corpus = [dictionary.doc2bow(sentence) for sentence in doc_word_list]
    # tfidf
    model = gensim.models.TfidfModel(corpus)
    corpus_tfidf = model[corpus]
    # onehotに変換し、ランクと関連付け
    rank_vector_dic_list = []
    accurate_doc2vec_count = 0
    for doc_index, doc in enumerate(corpus_tfidf):
        # doc -> word-ID:tfidf値の組合せのリスト
        onehot = np.zeros(len(dictionary.token2id))
        for word_index, val in doc:
            onehot[word_index] = val

        filename = train_corpus[doc_index].tags[0]
        # doc2vecのベクトルを生成
        with open(os.path.join(data_dir, '{}.txt'.format(filename)), 'r') as f:
            text = f.read()
            doc2vec_vector = get_doc2vec_vector.get_text_vector(text)
            # test model
            sims = get_doc2vec_vector.most_similar(doc2vec_vector)
            if sims[0][0] == filename:
                accurate_doc2vec_count += 1
            else:
                with open(os.path.join(data_dir, '{}.txt'.format(sims[0][0])), 'r') as ff:
                    infered_text = ff.read()
                print('++++++++++{}\n{}'.format(filename, text))
                print('----------{}\n{}\n'.format(sims[0][0], infered_text))

        #
        rank = os.path.splitext(filename)[0].split('_')[1]
        record = {'dv':doc2vec_vector, 'tv':onehot, 'r':rank, 'f':filename}
        rank_vector_dic_list.append(record)

    print('doc2vec model accuracy: {} / {}'.format(accurate_doc2vec_count, len(corpus_tfidf)))

    with open('spooky.pkl', 'wb') as f:
        pickle.dump(rank_vector_dic_list, f)

    pt = time.time() - st
    print("generate tfidf corpus completed, took {:.3f} milliseconds".format(pt))

    #generate_class_frequent_word_list(dictionary, rank_vector_dic_list)

    return corpus_tfidf


if __name__ == '__main__':
    print("reading corpus...")
    st = time.time()
    train_corpus = read_corpus_with_phrase(99999999)
    pt = time.time() - st
    print("reading corpus completed, took {:.3f} milliseconds".format(pt))

    print("training corpus...")
    st = time.time()
    model = generat_vector(train_corpus)
    pt = time.time() - st
    print("training corpus completed, took {:.3f} milliseconds".format(pt))
