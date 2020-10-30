import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import nltk, re
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

class Dataloader():
    def __init__(self, data_file):
        self.input_data = pd.DataFrame(data_file.readlines(), columns = ['Question'])
        self.input_data['QType'] = self.input_data.Question.apply(lambda x: x.split(' ', 1)[0])
        self.input_data['Question'] = self.input_data.Question.apply(lambda x: x.split(' ', 1)[1])
        self.input_data['QType-Coarse'] = self.input_data.QType.apply(lambda x: x.split(':')[0])
        self.input_data['QType-Fine'] = self.input_data.QType.apply(lambda x: x.split(':')[1])

        ##prepare small dataset
        self.small_data = self.input_data.sample(frac = 0.1)
        self.small_data.reset_index(drop=True, inplace=True)

        ##unique data
        self.unique_data = self.input_data.drop_duplicates(subset ="Question")
#       unique_data = unique_data.groupby('QType')
#       unique_data.apply(lambda x: x.sample(n = 100, replace = True))
#       unique_data.apply(lambda x: x.reset_index(drop=True, inplace=True))
        self.unique_data.reset_index(drop=True, inplace=True)

        self.wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
        self.common_dot_words = ['U.S.', 'St.', 'Mr.', 'Mrs.', 'D.C.']


    def text_clean(self, corpus, keep_list):
        '''
        Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)

        Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
        even after the cleaning process

        Output : Returns the cleaned text corpus

        '''
        cleaned_corpus = pd.Series()
        for row in corpus:
            qs = []
            for word in row.split():
                if word not in keep_list:
                    p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                    p1 = p1.lower()
                    qs.append(p1)
                else : qs.append(word)
            cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
        return cleaned_corpus

    def preprocess_sentence(self, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True, use_uniq = True):

        '''
        Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)

        Input :
        corpus: Text corpus on which pre-processing tasks will be performed
        'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should
                                                                  be performed or not
        'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
        Stemmer. 'snowball' corresponds to Snowball Stemmer

        Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together

        Output : Returns the processed text corpus

        '''
        if not use_uniq:
            input = self.input_data
        else:
            input = self.unique_data

        corpus =  pd.Series(input.Question.tolist()).astype(str)

        if cleaning:
            corpus = self.text_clean(corpus, self.common_dot_words)

        if remove_stopwords:
            stop = set(stopwords.words('english'))
            for word in self.wh_words:
                stop.remove(word)
            corpus = [[x for x in x.split() if x not in stop] for x in corpus]
        else :
            corpus = [[x for x in x.split()] for x in corpus]

        if lemmatization:
            lem = WordNetLemmatizer()
            corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]

        if stemming:
            if stem_type == 'snowball':
                stemmer = SnowballStemmer(language = 'english')
                corpus = [[stemmer.stem(x) for x in x] for x in corpus]
            else :
                stemmer = PorterStemmer()
                corpus = [[stemmer.stem(x) for x in x] for x in corpus]

        corpus = [' '.join(x) for x in corpus]

        return corpus, pd.Series(input['QType'].tolist()).astype(str), pd.Series(input['QType-Coarse'].tolist()).astype(str)

    def label_create(self, use_uniq = False):
        '''
        Create Labels for each group
        '''
        if not use_uniq:
            input = self.input_data
        else:
            input = self.unique_data

        le = LabelEncoder()
        le.fit(pd.Series(input['QType'].tolist()).values)
        input['QType'] = le.transform(input.QType.values)

        le2 = LabelEncoder()
        le2.fit(pd.Series(input['QType-Coarse'].tolist()).values)
        input['QType-Coarse'] = le2.transform(input['QType-Coarse'].values)

        le3 = LabelEncoder()
        le3.fit(pd.Series(input['QType-Fine'].tolist()).values)
        input['QType-Fine'] = le3.transform(input['QType-Fine'].values)

        return input['QType'].values, input['QType-Coarse'].values
