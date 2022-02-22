import os
import pickle
import argparse

import pandas as pd
from tqdm import tqdm 

from settings import (EXPERIMENTS_BASE_DIR, COMBINATIONS_BASE_PATH)

from generate_embeddings import generate_new_embeddings
from services_logreg import (get_vectors_name, get_Xy_data, train_lr_bin, train_lr_multi)
from services_metrics_with_multi import (get_vectorname, load_lr_models, compute_csv_default)
from services_embeddings import get_embeddings_corpus

parser = argparse.ArgumentParser(prog='MAIN', description="Probing some LM's")
parser.add_argument('-e', '--experiment_name', type=str, help='set here your experiment name', default='default_experiment')
parser.add_argument('-c', '--compute_raw_embeddings', type=int, help='if you want to compute raw vectors experiment set 1 else 0', default=0)

parser.add_argument('-ct', '--compute_raw_embeddings_train', type=int, help='if you want to compute train raw vectors experiment set 1 else 0', default=1)
parser.add_argument('-cte', '--compute_raw_embeddings_test', type=int, help='if you want to compute test raw vectors experiment set 1 else 0', default=1)
parser.add_argument('-cv', '--compute_raw_embeddings_valid', type=int, help='if you want to compute valid raw vectors experiment set 1 else 0', default=1)

args = parser.parse_args()

experiment_name = args.experiment_name

experiment_path = os.path.join(EXPERIMENTS_BASE_DIR, experiment_name)
vectors_folder = os.path.join(experiment_path, 'vectors')
logreg_folders = os.path.join(experiment_path, 'logreg_models')
val_results_folders = os.path.join(experiment_path, 'val_results')
data_type = 'train-val-test' # TODO add train-val-test-nb inside data/ folder 

if args.compute_raw_embeddings:
    attentions_types = [1, 1, 1, 1, 1, 1]
    if args.compute_raw_embeddings_train:
        get_embeddings_corpus(vectors_folder, data_type, 'train', attentions_types)
    if args.compute_raw_embeddings_test:
        get_embeddings_corpus(vectors_folder, data_type, 'test', attentions_types)
    if args.compute_raw_embeddings_valid:
        get_embeddings_corpus(vectors_folder, data_type, 'valid', attentions_types)

vectors_with_six_attentions = os.path.join(vectors_folder, 'valid_h-r_r-t_h-t_r-h_t-r_t-h.pkl')

os.makedirs(logreg_folders, exist_ok=True)
os.makedirs(val_results_folders, exist_ok=True)

combinations = pd.read_excel(COMBINATIONS_BASE_PATH)
combinations = combinations.dropna().astype(int).drop_duplicates().loc[2:,:].reset_index(drop=True)
combinations = combinations.iloc[3:,:6].drop_duplicates().reset_index(drop=True)

val_data = pd.read_csv(f'../data/{data_type}/valid.csv')

with open(vectors_with_six_attentions, 'rb') as file:
    full_embeddings = pickle.load(file)

for idx, comb in tqdm(enumerate(combinations.itertuples()), total=len(combinations)):
    # set params
    attentions_types = list(comb)[1:]
    print('Combinations: ', attentions_types)
    
    if idx < 63:
        continue
    
    # generate new embeddings
    generate_new_embeddings('train', vectors_folder, attentions_types, False, False)    
    generate_new_embeddings('test', vectors_folder, attentions_types, False, False)    
    
    # train models
    vectors_name = get_vectors_name(attentions_types, False, False)
    X_train, y_train = get_Xy_data('train', vectors_folder, vectors_name)
    X_test, y_test   = get_Xy_data('test',  vectors_folder, vectors_name)
        
    train_lr_bin(X_train,   y_train, X_test, y_test, vectors_name, logreg_folders)
    train_lr_multi(X_train, y_train, X_test, y_test, vectors_name, logreg_folders)
    
    # compute metrics
    vectorname = get_vectorname(attentions_types, False, False)    
    lr_bin, lr_multi = load_lr_models(logreg_folders, vectorname)
    compute_csv_default(val_data, lr_bin, lr_multi, attentions_types, False, False, full_embeddings, filename=f'res_{vectorname}', folder=val_results_folders)
