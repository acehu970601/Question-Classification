from data_loader import Dataloader
from feature_generator import FeatureGenerator
from utils import split_data, plot_learning_curve, aucroc_scorer, plot_validation_curve, data_augmentation
import argparse
import nltk
import numpy as np
from scipy.sparse import hstack
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score

if __name__ == '__main__':
    ## parser and prpare downloading
    parser = argparse.ArgumentParser(description='Put parameters here')
    parser.add_argument('--datapath', type=str, default='training.data', dest='filename', help='input file path')
    parser.add_argument('--label_type', type=str, default='coarse', dest='label_type', help='classify fine types / coarse types')
    parser.add_argument('--model_name', type=str, default='SVC', dest='model_name', help='Choose the desired model')
    parser.add_argument('--use_uniq_only', action='store_true', dest='uniq_data', help='Drop all duplicate data')
    parser.add_argument('--balance_class', type=str, default='balanced', dest='balance_class', help='Use blanced weights for classes')
    parser.add_argument('--use_stemmer', action='store_true', dest='use_stem', help='Use stemming during preprocessing')
    parser.add_argument('--use_lemma', action='store_true', dest='use_lemma', help='Use lemmatization during preprocessing')
    parser.add_argument('--max_iters', type=int, default = 5000, dest='max_iters', help='Choose the desired max iters')
    parser.add_argument('--plot', action='store_true', dest='plot_learning_curve', help='Plot learning curve of the model or not')
    parser.add_argument('--finetune', action='store_true', dest='plot_validation_curve', help='Plot validation curve of the model to find best parameter')
    parser.add_argument('--augmentdata', action='store_true', dest='data_augmentation', help='Enabling data augmentation')
    args = parser.parse_args()
    nltk.download('stopwords')
    nltk.download('wordnet')
    #####

    ## Load , preprocess sentestemming = args.use_stem, nce and create labels
    data_file = open(args.filename, 'r')
    data_loader = Dataloader(data_file)
    cleaned_sentence, fine_type, coarse_type = data_loader.preprocess_sentence(remove_stopwords = True, stemming = args.use_stem, lemmatization = args.use_lemma, use_uniq = args.uniq_data)
    fine_label, coarse_label = data_loader.label_create(use_uniq = args.uniq_data)

    ## Generate features from sentences
    feature_generator = FeatureGenerator(cleaned_sentence)
    ner_ft, lemma_ft, tag_ft, dep_ft, shape_ft = feature_generator.transform()

    # features = hstack([ner_ft, lemma_ft, tag_ft, dep_ft, shape_ft]).tocsr()
    features = hstack([ner_ft, lemma_ft, tag_ft]).tocsr()
    if args.data_augmentation:
        if args.label_type == 'coarse':
            features, coarse_label = data_augmentation(features, coarse_label)
        else:
            features, fine_label = data_augmentation(features, fine_label)
    train_feat, train_labels, val_feat, val_labels, test_feat, test_labels = split_data(features, fine_label, coarse_label, args.label_type)

    ## train the model and output scores
    if args.model_name == 'SVC':
        model = svm.SVC(max_iter = args.max_iters, class_weight = args.balance_class, probability = True, C = 6.0)
    elif args.model_name == 'LinearSVC':
        model = svm.LinearSVC(max_iter = args.max_iters, class_weight = args.balance_class)
    elif args.model_name == 'RandomForest':
        model = RandomForestClassifier(class_weight = args.balance_class)
    elif args.model_name == 'SGDClassifier':
        model = linear_model.SGDClassifier()

    model.fit(train_feat, train_labels)
    if args.model_name == 'SVC':
        aucroc_val, aucroc_test = aucroc_scorer(model, val_feat, val_labels), aucroc_scorer(model, test_feat, test_labels)
        print (aucroc_val, aucroc_test)
    else:
        pred,test_pred = model.predict(val_feat), model.predict(test_feat)
        scores, test_scores = accuracy_score(val_labels, pred), accuracy_score(test_labels, test_pred)
        print (scores, test_scores)

    ## metrics output and plotting
    if args.plot_learning_curve:
        plot_learning_curve(model, args.model_name, train_feat, train_labels, ylim = (0.1, 1.0), cv=2)
    if args.plot_validation_curve:
        plot_validation_curve(model, args.model_name, train_feat, train_labels, param_name = 'C')
