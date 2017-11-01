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

# data_dir = '/var/data/wiki_files'
# data_dir = '/home/satoshi/data/tech_chem'
#data_dir = '/home/satoshi/dev/ttp/textfiles'
data_dir = '/home/satoshi/dev/kaggle/spooky/data/text'
min_word_count = 0
model_save_file = 'spooky.model'
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


def read_corpus(max_data_count=10000):
    train_corpus = []
    #if os.path.exists(train_corpus_pickle):
        #with open(train_corpus_pickle, mode='rb') as f:
            #train_corpus = pickle.load(f)
        #return train_corpus

    bs = BuildSentence(mt)

    corpus_cnt = 0
    titles = []
    sentences = []
    train_cnt = 0
    test_cnt = 0
    for filename in [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]:
        if corpus_cnt >= max_data_count:
            break

        try:
            with open(os.path.join(data_dir, filename)) as f:
                title = filename.replace(".txt", "")
                content = f.read()
            #lang = detect(content)
            lang = 'en'
            if lang != 'ja' and lang != 'en':
                print('{}: {}'.format(lang, title))
                continue
            if lang == 'en':
                content = bs.nlp_processing(content)
            else:
                #content = gensim.utils.simple_preprocess(mt.parse(content), min_len=1)
                #content = mt.parse(content).split(' ')
                content = bs.do_mecab(content)
            titles.append(title)
            sentences.append(content + content)
            corpus_cnt += 1
            if len(title.split('_')[1]) > 0:
                train_cnt += 1
            else:
                test_cnt += 1
        except:
            pass

    print('train: {}, test: {}'.format(train_cnt, test_cnt))

    #m = gensim.models.Phrases(min_count=5)
    #m.add_vocab(sentences)
    # sentences_1 = []
    # for i, s in enumerate(m[sentences]):
    #    new_s = build_sentence(s, stop_words=[])
    #    sentences_1.append(new_s)

    # m.add_vocab(sentences_1)
    # for i, s in enumerate(m[sentences_1]):
    #sentences = bs.build_sentence_list(m, sentences)
    work_min = 9999
    for i, s in enumerate(sentences):
        # new_s = bs.build_sentence(s)
        tagged_doc = gensim.models.doc2vec.TaggedDocument(s, [titles[i]])
        if len(tagged_doc.words) < work_min:
            work_min = len(tagged_doc.words)
            print(s, titles[i])
        if len(tagged_doc.words) < min_word_count:
            continue
        train_corpus.append(tagged_doc)

    print(work_min)
    #m.save(phrase_file)

    return train_corpus


def train_model(train_corpus):
    model = gensim.models.doc2vec.Doc2Vec(size=vector_size, window=window_size, min_count=min_count, iter=iter,
                                          workers=workers, hs=1, negative=0, dm=1, dm_concat=0)
    st = time.time()
    model.build_vocab(train_corpus, keep_raw_vocab=True)
    pt = time.time() - st
    print("  build_vocab() completed, took {:.3f} milliseconds".format(pt))
    st = time.time()
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    pt = time.time() - st
    print("  train() completed, took {:.3f} milliseconds".format(pt))
    st = time.time()
    model.save(model_save_file)
    pt = time.time() - st
    print("  save() completed, took {:.3f} milliseconds".format(pt))

    return model


def load_model():
    model = gensim.models.doc2vec.Doc2Vec.load(model_save_file)
    return model


def test_model(model, train_corpus):
    ranks = []
    for i, corpus in enumerate(train_corpus):
        title = corpus.tags[0]
        inferred_vector = model.infer_vector(corpus.words)
        sims = model.docvecs.most_similar([inferred_vector], topn=3)
        if title == sims[0][0]:
            ranks.append(0)
        else:
            print("{} {}".format(i, title))
            for j, t in enumerate(sims):
                print("  {}: {}, {:.2f}".format(j + 1, t[0], t[1]))
            ranks.append(1)

    ranks_counter = collections.Counter(ranks)
    accuracy = float(ranks_counter[0]) / float(len(train_corpus))
    print("accuracy: {:.2f} {:d} / {:d}".format(accuracy, ranks_counter[0], len(train_corpus)))


if __name__ == '__main__':
    model_save_file = sys.argv[1] + '.model'
    train_corpus_pickle = sys.argv[1] + '.pkl'
    phrase_file = sys.argv[1] + '.phrase'
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]

    if os.path.exists(model_save_file):
        model = load_model()
        filename = "test.txt"
        test_doc = []
        with open(filename) as f:
            content = ""
            for l in f:
                content += l
            # sliced_content = content[int(len(content)*0.3):int(len(content)*0.5)]
            # sliced_content = "日立製作所は、その創業の精神である和、誠、開拓者精神をさらに高揚させ、日立人としての誇りを堅持し、優れた自主技術・製品の開発を通じて社会に貢献することを基本理念とする。 あわせて、当社は、企業が社会の一員であることを深く認識し、公正かつ透明な企業行動に徹するとともに、環境との調和、積極的な社会貢献活動を通じ、良識ある市民として真に豊かな社会の実現に尽力する。"
            sliced_content = content
            print(sliced_content)
            test_doc = gensim.utils.simple_preprocess(mt.parse(sliced_content))
        inferred_vector = model.infer_vector(test_doc)
        sims = model.docvecs.most_similar([inferred_vector], topn=10)
        for j, t in enumerate(sims):
            print("  {}: {}, {:.2f}".format(j + 1, t[0], t[1]))
    else:
        print("reading corpus...")
        st = time.time()
        # train_corpus = read_corpus(99999999)
        train_corpus = read_corpus(99999999)
        pt = time.time() - st
        print("reading corpus completed, took {:.3f} milliseconds".format(pt))

        print("training corpus...")
        st = time.time()
        model = train_model(train_corpus)
        pt = time.time() - st
        print("training corpus completed, took {:.3f} milliseconds".format(pt))
