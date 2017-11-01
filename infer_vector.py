#! -*- coding: UTF-8 -*-
import MeCab
import gensim
from doc2vec_utils import BuildSentence
from doc2vec_utils import BuildSentence


class GetDoc2vecVector:

    def __init__(self, model_file):
        self.model = gensim.models.doc2vec.Doc2Vec.load(model_file)
        self.infer_vector_steps = 15
        self.bs = BuildSentence(None)

    def get_text_vector(self, text):
        #text = gensim.utils.simple_preprocess(text, min_len=1)
        text = self.bs.nlp_processing(text)

        vec = self.model.infer_vector(text, steps=self.infer_vector_steps)

        return vec

    def most_similar(self, vec):
        return self.model.docvecs.most_similar(positive=[vec])




if __name__ == '__main__':
    pass
