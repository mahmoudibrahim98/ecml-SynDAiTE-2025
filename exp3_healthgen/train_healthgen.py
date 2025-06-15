import sys
dir = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff'
sys.path.append(dir)

import os
import numpy as np
import torch 
import sys

import data_access.base_loader as base_loader
import data_access.ricu_loader as ricu_loader
import datetime
import wandb
import ast
import logging
import json
import timeautodiff.processing_simple as processing
import timeautodiff.helper_simple as tdf_helper
import timeautodiff.timeautodiff_v4_efficient_simple as timeautodiff
import torch

def main(task_name, data_name):

    # splitting parameters
    train_fraction = 0.45
    val_fraction = 0.05
    oracle_fraction = 0
    oracle_min = 100
    intersectional_min_threshold = 100
    intersectional_max_threshold = 1000



    static_var = 'ethnicity'
    features = None
    ricu_dataset_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/0_Synthetic-data-experiments/real_data/raw/{task_name}/{data_name}'
    # processed_output_path = f'../../real_data/processed/{task_name}/{data_name}'
    # intermed_output_path = f'../../real_data/intermed/{task_name}/{data_name}'
    # processed_data_timestamp = '20250113132215'
    processed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0a_ecml_conditional_healthgen_generation/outputs/{task_name}/{data_name}/processed/'
    intermed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0a_ecml_conditional_healthgen_generation/outputs/{task_name}/{data_name}/intermed/'
    seed = 0

    simple_imputation = True
    mode = 'processed'
    data_timestamps = {'los_24':{'eicu': '20250502191500' , 'mimic': '20250528125335'},
                    'mortality24':{'eicu': '20250502191700' , 'mimic': '20250528125252'} }
    processed_data_timestamp = data_timestamps[task_name][data_name]

    intermed_data_timestamp = None

    standardize = False
    save_intermed_data = True
    save_processed_data = True
    split = True
    stratify =  False
    intersectional = False

    if split == False:
        split_text = 'No Split'
    else:
        split_text = 'Split'
    data_params = {
        'processed_data_timestamp':processed_data_timestamp,
        'task_name': task_name,
        'data_name': data_name,
        'train_fraction': train_fraction,
        'val_fraction': val_fraction,
        'test_fraction': 1 - train_fraction - val_fraction,
        'oracle_fraction': oracle_fraction,
        'oracle_min': oracle_min,
        'intersectional_min_threshold': intersectional_min_threshold,
        'intersectional_max_threshold': intersectional_max_threshold,
        'split': split_text,
        'standardize' : standardize,
    }

    loader = ricu_loader.RicuLoader(seed, task_name, data_name,static_var,ricu_dataset_path,simple_imputation,
                                        features, processed_output_path,intermed_output_path)





    X_dict_tf, y_dict, static = loader.get_data(
        mode='processed', 
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        oracle_fraction=oracle_fraction,
        oracle_min=oracle_min,
        intersectional_min_threshold=intersectional_min_threshold,
        intersectional_max_threshold=intersectional_max_threshold,
        stratify=stratify,
        intersectional=intersectional,
        save_intermed_data=False,
        save_processed_data=False,
        demographics_to_stratify_on = ['age_group','ethnicity','gender'],
        processed_timestamp=processed_data_timestamp
    )
        
    if not isinstance(X_dict_tf, dict):
        X_dict_tf = {file: X_dict_tf[file] for file in X_dict_tf.files}
        y_dict = {file: y_dict[file] for file in y_dict.files}

    # data_params = {
    #     'processed_data_timestamp':processed_data_timestamp,
    #     'task_name': task_name,
    #     'data_name': data_name,
    #     'train_fraction': train_fraction,
    #     'val_fraction': val_fraction,
    #     'test_fraction': test_fraction,
    #     'oracle_fraction': oracle_fraction,
    #     'oracle_min': oracle_min,
    #     'intersectional_min_threshold': intersectional_min_threshold,
    #     'intersectional_max_threshold': intersectional_max_threshold,
    #     'split': split_text,
    #     'standardize' : standardize,
    # }
    X_dict_tf.keys()


    # most_important_features = [19, 27, 17, 35, 22, 44, 42, 43, 37, 26]
    X_train = X_dict_tf['X_imputed_train'][:,:,:]
    X_test = X_dict_tf['X_imputed_test'][:,:,:]
    X_val = X_dict_tf['X_imputed_val'][:,:,:]

    m_train = X_dict_tf['m_train'][:,:,:]
    m_test = X_dict_tf['m_test'][:,:,:]
    m_val = X_dict_tf['m_val'][:,:,:]

    feature_names = X_dict_tf['feature_names'][:]
    y_train = y_dict['y_train'][:]
    y_test = y_dict['y_test'][:]
    y_val = y_dict['y_val'][:]


    static_feature_to_include = ['ethnicity','gender','age_group']
    static_features_to_include_indices = sorted([y_dict['feature_names'].tolist().index(include)  for include in static_feature_to_include])
    c_train = y_dict['c_train'][:,static_features_to_include_indices]
    c_test = y_dict['c_test'][:,static_features_to_include_indices]
    c_val = y_dict['c_val'][:,static_features_to_include_indices]

    cond_names = y_dict['feature_names'][static_features_to_include_indices]

    # TODO into helpers


    top10_important_features = [19, 27, 17, 35, 22, 44, 42, 43, 37, 26]
    top3_important_features = [44,42,43]
    top6_important_features = [42, 22, 27, 35, 43, 17]

    important_features_names = X_dict_tf['feature_names'][top10_important_features]
    important_features_names

    X_train_10 = processing.normalize_and_reshape(X_train)
    X_train_10 = X_train_10[:,:,top10_important_features]

    print('Shape of X train:', X_train.shape)
    print('Shape of X test:', X_test.shape)
    print('Shape of X val:', X_val.shape)

    print('Shape of y train:', y_train.shape)
    print('Shape of y test:', y_test.shape)
    print('Shape of y val:', y_val.shape)

    print('Shape of c train:', c_train.shape)
    print('Shape of c test:', c_test.shape)
    print('Shape of c val:', c_val.shape)




    metadata = f"{data_name}_{task_name}"

    process_data = True
    load_data = False
    train_models = True
    train_auto = True
    train_diff = True
    load_model = False
    # processed_data_timestamp = '20241203_130537_10features'

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
                                                        # Prepare Data for Training #
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    model_version = 'v4_efficient_simple'




    choose_features_for_compatibility = [22,42,35,27]
    # prorcess data for training of generators
    processed_X, processed_y, processed_c, time_info = processing.process_data_for_synthesizer(X_train, y_train, c_train, choose_features_for_compatibility)
    cond = torch.concatenate((processed_c, processed_y), axis=2)
    response = processed_X
    response = response.float()
    time_info = time_info.float()


    X_healthgen = [
        processed_X[:,:,0], # heart rate    
        processed_X[:,:,1], # respiration
        processed_X[:,:,2], # o2 saturation
        processed_X[:,:,3], # mean arterial pressure
        ]
    X_healthgen = np.stack(X_healthgen, axis=2)
    X_healthgen = np.transpose(X_healthgen, (0, 2, 1))
    X_healthgen = torch.tensor(X_healthgen).float()

    m_healthgen = [ np.transpose(m_train, (0, 2, 1))[:,:,22], # heart rate mask
        np.transpose(m_train, (0, 2, 1))[:,:,42], # respiration mask
        np.transpose(m_train, (0, 2, 1))[:,:,35], # o2 saturation mask
        np.transpose(m_train, (0, 2, 1))[:,:,27], # mean arterial pressure mask
    ]
    m_healthgen = np.stack(m_healthgen, axis=2)
    m_healthgen = np.transpose(m_healthgen, (0, 2, 1))




    dir = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/0_Synthetic-data-experiments'
    sys.path.append(dir)
    # from healthgen.generation import VAEGenModel, MultiVAEGenModel, SRNNGenModel, KVAEGenModel, KVAEMissGenModel, HealthGenModel
    from healthgen import HealthGenModel
    from sklearn.model_selection import train_test_split

    # Split for internal tarining and validation of gen model
    indices = np.arange(len(X_healthgen))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)

    healthgen_X = {'X_train': X_healthgen[train_idx], 'X_val': X_healthgen[test_idx],
                'm_train': m_healthgen[train_idx], 'm_val': m_healthgen[test_idx]}
    healthgen_y = {'y_train': y_train[train_idx], 'y_val': y_train[test_idx],
                'feature_names': cond_names,
                'c_train': c_train[train_idx], 'c_val': c_train[test_idx]}
    static_vars = cond_names
    if len(static_vars) > 0:
        cond_static = True
    else:
        cond_static = False
    healthgen_config = {
        'seed' : seed,
        'data': data_name,
        'task_name': task_name,
        'data_mode' : 'feats_mask',
        'cond_static': cond_static,
        'static_vars' : static_vars,
        'gen_model_name': 'healthgen',
        'metadata' : f'{data_name}_{task_name}_outcome',
        'x_dim' :  healthgen_X['X_train'].shape[1],
        'y_dim' : len(static_vars) + 1,
        'z_dim' : 16, #Latent space dimension. , 
        'seq_len' : healthgen_X['X_train'].shape[2], #Length of the input sequences.
        'out_path' : os.getcwd(),
        'gen_epochs' : 100,
        'use_wandb' : False
        
    }
    healthgen_model =  HealthGenModel(healthgen_config)


    config = dict(
        seed = seed,
        dataset = data_name,
        model = healthgen_model.gen_model,
        pred_task = task_name,
        
    )

    use_cuda = torch.cuda.is_available()

    # wandb.init(
    #     project = 'time_series',
    #     config = config,
    #     name = f'{healthgen_model.gen_model}_{data_name}_{task_name}_outcome',
    #     reinit="create_new"
    # )

    healthgen_model.train_model(healthgen_X,healthgen_y)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TimeAutoDiff model with regularization weights')
    parser.add_argument('--task_name', type=str, required=True, help='task_name')
    parser.add_argument('--data_name', type=str, required=True, help='data_name')

    args = parser.parse_args()
    
    main(task_name=args.task_name, 
         data_name=args.data_name)
