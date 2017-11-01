# -*- coding: utf-8 -*-
import re
import nltk

class BuildSentence:
    def __init__(self, mt):
        self.stop_words = {}
        self.asciiReg = re.compile(r'^[!-~]+$')
        self.alpha_regexp = re.compile(r'[^a-zA-Z_]')

        self.mt = mt

        self.stopwords_set = nltk.corpus.stopwords.words('english')
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stemmer = nltk.stem.LancasterStemmer()

    def nlp_processing(self, dialog):
        work = self.stemmer.stem(dialog)
        work = self.lemmatizer.lemmatize(work)
        work = nltk.tokenize.word_tokenize(work)

        return work
        #return [w for w in work if len(w) > 2 and w not in self.stopwords_set]

    def do_mecab(self, original):
        word_list = self.mt.parse(original)
        words_in_content = []
        for line in word_list.split('\n'):
            split_by_tab = line.split('\t')
            if len(split_by_tab) >= 1:
                words_in_content.append(split_by_tab[0])
                if len(split_by_tab) >= 2:
                    # 助詞、助動詞、記号、数はあとで消すためにリストに追加
                    split_by_connma = split_by_tab[1].split(',')
                    if split_by_tab[0] not in self.stop_words:
                        if split_by_connma[0] == '助詞':
                            self.stop_words[split_by_tab[0]] = '助詞'
                        if split_by_connma[0] == '助動詞':
                            self.stop_words[split_by_tab[0]] = '助動詞'
                        if split_by_connma[0] == '記号':
                            self.stop_words[split_by_tab[0]] = '記号'
                        if (split_by_connma[0] == '名詞' and split_by_connma[1] == '数'):
                            self.stop_words[split_by_tab[0]] = '数'

        return words_in_content


    def build_sentence_list(self, phrase, sentences):
        # Phrasesを複数回行う
        phrase_time = 3
        return_sentences = []
        for i in range(phrase_time):
            sentences = phrase[sentences]
            if i == (phrase_time-1):
                for i, s in enumerate(sentences):
                    new_s = self.build_sentence(s)
                    return_sentences.append(new_s)

        return return_sentences

    def build_sentence(self, sentence):
        # sentenseはPhrase処理済みのもの
        new_s = []
        for w in sentence:
            if self.asciiReg.search(w) != None:
                # 英単語
                if len(w) == 1:
                    # 1文字は削除、gensimのsimple_preprocess()を参考
                    w = ''
                else:
                    # アルファベットと数値だけにして小文字
                    w = self.alpha_regexp.sub('', w).lower()
                    if '_' in w:
                        w_split = w.split('_')
                        w = ''
                        for ww in w_split:
                            if len(ww) > 1:
                                if ww not in self.stop_words or self.stop_words[ww] != '記号':
                                    w = w + ww + ' '
                w = w.strip()
                if len(w) > 0:
                    if w not in self.stop_words or self.stop_words[w] != '記号':
                        new_s.append(w)
            else:
                # 日本語
                if '_' in w:
                    # Phrasesで連結された語をひとつづつチェックする
                    # 連結された語の中のストップワードは連結しない
                    w_split = w.split('_')
                    w = ''
                    before_lang = ''
                    for ww in w_split:
                        if ww in self.stop_words:
                            # ストップワードの場合はくっつけない
                            if len(w) > 1:
                                # 前の語を登録
                                new_s.append(w)
                                w = ''
                                before_lang = '' # くっつけないので前の語の言語もリセット
                            if self.stop_words[ww] != '記号':
                                # 記号の場合はドロップ、それ以外はくっつけず単独で登録
                                new_s.append(ww)
                        else:
                            # ストップワードでないのでくっつける
                            if self.asciiReg.search(ww) != None:
                                # 英語
                                # アルファベットと数値だけにして小文字
                                if len(ww) > 1:
                                    ww = self.alpha_regexp.sub('', ww).lower()
                                    if before_lang == 'e':
                                        w = w + ' ' + ww
                                    else:
                                        # 前の語が英語でない場合はスペースをおかない
                                        w += ww
                                before_lang = 'e'
                            else:
                                # 日本語
                                w += ww
                                before_lang = 'j'

                    if len(w) > 1:
                        new_s.append(w)
                else:
                    if self.asciiReg.search(w) != None:
                        # 英語
                        # 1文字は削除、gensimのsimple_preprocess()を参考
                        if len(w) == 1:
                            w = ''
                        else:
                            # アルファベットと数値だけにして小文字
                            w = self.alpha_regexp.sub('', w).lower().strip()

                    if len(w) > 0:
                        if w not in self.stop_words or self.stop_words[w] != '記号':
                            new_s.append(w)

        return new_s
