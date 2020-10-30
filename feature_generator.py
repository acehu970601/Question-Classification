from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
import spacy

class FeatureGenerator():
    def __init__(self, corpus):
        self.nlp = spacy.load('en_core_web_sm')
        self.ner, self.lemma, self.tag, self.dep, self.shape = self.transform_features(corpus)
        self.count_vec_ner = CountVectorizer(ngram_range=(1, 2)).fit(self.ner)
        self.count_vec_lemma = CountVectorizer(ngram_range=(1, 2)).fit(self.lemma)
        self.count_vec_tag = CountVectorizer(ngram_range=(1, 2)).fit(self.tag)
        self.count_vec_dep = CountVectorizer(ngram_range=(1, 2)).fit(self.dep)
        self.count_vec_shape = CountVectorizer(ngram_range=(1, 2)).fit(self.shape)

    def transform(self):
        '''
        Converting the features into vectors using CountVectorizer
        '''
        ner_ft = self.count_vec_ner.transform(self.ner)
        lemma_ft = self.count_vec_lemma.transform(self.lemma)
        tag_ft = self.count_vec_tag.transform(self.tag)
        dep_ft = self.count_vec_dep.transform(self.dep)
        shape_ft = self.count_vec_shape.transform(self.shape)
        return ner_ft, lemma_ft, tag_ft, dep_ft, shape_ft

    def transform_features(self, corpus):
        '''
        Creating list of Named Entitites, Lemmas, POS Tags, Syntactic Dependency Relation and Orthographic Features using shapes
        '''
        all_ner = []
        all_lemma = []
        all_tag = []
        all_dep = []
        all_shape = []
        for sentence in corpus:
            doc = self.nlp(sentence)
            present_lemma = []
            present_tag = []
            present_dep = []
            present_shape = []
            present_ner = []
            for token in doc:
                present_lemma.append(token.lemma_)
                present_tag.append(token.tag_)
                present_dep.append(token.dep_)
                present_shape.append(token.shape_)
            all_lemma.append(" ".join(present_lemma))
            all_tag.append(" ".join(present_tag))
            all_dep.append(" ".join(present_dep))
            all_shape.append(" ".join(present_shape))
            for ent in doc.ents:
                present_ner.append(ent.label_)
            all_ner.append(" ".join(present_ner))
        return all_ner, all_lemma, all_tag, all_dep, all_shape
