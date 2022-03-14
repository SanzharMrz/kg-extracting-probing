import os
import pickle

import itertools
import pandas as pd
from tqdm import tqdm 

from settings import EXPERIMENTS_BASE_DIR

from generate_embeddings import generate_new_embeddings
from services_logreg import (get_vectors_name, get_Xy_data, train_lr_bin, train_lr_multi)
from services_metrics_with_multi import (get_vectorname, load_lr_models, compute_csv_default, compute_csv)
from services_embeddings import get_embeddings_corpus

from utils_nltk import get_args


args = get_args()
experiment_name = args.experiment_name
stages = args.stages
ner = None

if args.ner:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner)


experiment_path = os.path.join(EXPERIMENTS_BASE_DIR, experiment_name)
vectors_folder = os.path.join(experiment_path, 'vectors')
logreg_folders = os.path.join(experiment_path, 'logreg_models')
val_results_folders = os.path.join(experiment_path, 'val_results')
data_type = 'train-val-test' # TODO add train-val-test-nb inside data/ folder 

os.makedirs(vectors_folder, exist_ok=True)
os.makedirs(logreg_folders, exist_ok=True)
os.makedirs(val_results_folders, exist_ok=True)

if args.compute_raw_embeddings:
    attentions_types = [1, 1, 1, 1, 1, 1]
    if args.compute_raw_embeddings_train:
        get_embeddings_corpus(vectors_folder, data_type, 'train', attentions_types)
    if args.compute_raw_embeddings_test:
        get_embeddings_corpus(vectors_folder, data_type, 'test', attentions_types)
    if args.compute_raw_embeddings_valid:
        get_embeddings_corpus(vectors_folder, data_type, 'valid', attentions_types, ner=ner)

vectors_with_six_attentions = os.path.join(vectors_folder, 'valid_h-r_r-t_h-t_r-h_t-r_t-h.pkl')

combinations = [p for p in itertools.product([1, 0], repeat=6)][:-1]

val_data = pd.read_csv(f'../data/{data_type}/valid.csv')

with open(vectors_with_six_attentions, 'rb') as file:
    full_embeddings = pickle.load(file)

for idx, comb in tqdm(enumerate(combinations), total=len(combinations)):
    # set params
    print('Combinations: ', comb)
    
    if idx > 0:
        break
    
    # generate new embeddings
    if stages[0]:
        generate_new_embeddings('train', vectors_folder, comb, False, False)    
        generate_new_embeddings('test', vectors_folder, comb, False, False)    
    
    # train models
    if stages[1]:
        vectors_name = get_vectors_name(comb, False, False)
        X_train, y_train = get_Xy_data('train', vectors_folder, vectors_name)
        X_test, y_test   = get_Xy_data('test',  vectors_folder, vectors_name)
        train_lr_bin(X_train,   y_train, X_test, y_test, vectors_name, logreg_folders)
        train_lr_multi(X_train, y_train, X_test, y_test, vectors_name, logreg_folders)
    
    # compute metrics
    if stages[2]:
        vectorname = get_vectorname(comb, False, False)    
        lr_bin, lr_multi = load_lr_models(logreg_folders, vectorname)
        # compute_csv_default(val_data, lr_bin, lr_multi, comb, False, False, full_embeddings, filename=f'res_{vectorname}', folder=val_results_folders)
        compute_csv(val_data, lr_bin, lr_multi, comb, False, False, full_embeddings, filename=f'res_{vectorname}', folder=val_results_folders)
