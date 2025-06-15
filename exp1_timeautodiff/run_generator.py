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

import importlib
import timeautodiff.timeautodiff_v4_efficient_simple as timeautodiff
importlib.reload(timeautodiff)
import timeautodiff.processing_simple as processing
import timeautodiff.helper_simple as tdf_helper
import timeautodiff.timeautodiff_v4_efficient_simple as timeautodiff
import evaluation_framework.vis as vis
import torch
import pandas as pd

def main(task_name, data_name,exp_id,
         auto_mmd_weight, auto_consistency_weight, 
         diff_mmd_weight, diff_consistency_weight):
    os.chdir('/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0_ecml_conditional_autodiff_generation/scripts_training_and_evaluating_only_consis')
    print(os.getcwd())
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
    processed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0_ecml_conditional_autodiff_generation/outputs/{task_name}/{data_name}/processed/'
    intermed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0_ecml_conditional_autodiff_generation/outputs/{task_name}/{data_name}/intermed/'
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
    models = []
    for i in range(5):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f'Training model {i+1} of 5...')
        model_version = 'v4_efficient_simple'

            
        EXP_PATH = os.path.join(os.getcwd(), 'outputs')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_model = 'TimeAutoDiff'
        output_dir = f'outputs/{task_name}/{data_name}/{gen_model}/{timestamp}_{len(important_features_names)}features_{exp_id}'
        os.makedirs(output_dir, exist_ok=True)
        numerical_processing = 'normalize'
        diff_timestamp = f'{timestamp}_{len(important_features_names)}features_{exp_id}'

        choose_features_for_compatibility = [22,42,35,27]
        # prorcess data for training of generators
        processed_X, processed_y, processed_c, time_info = processing.process_data_for_synthesizer(X_train, y_train, c_train, choose_features_for_compatibility)
        REAL_COND = torch.concatenate((processed_c, processed_y), axis=2)
        time_info_real = time_info.float()


        arrays_to_concat = [
            processed_X[:,:,0], # heart rate    
            np.transpose(m_train, (0, 2, 1))[:,:,22], # heart rate mask
            processed_X[:,:,1], # respiration
            np.transpose(m_train, (0, 2, 1))[:,:,42], # respiration mask
            processed_X[:,:,2], # o2 saturation
            np.transpose(m_train, (0, 2, 1))[:,:,35], # o2 saturation mask
            processed_X[:,:,3], # mean arterial pressure
            np.transpose(m_train, (0, 2, 1))[:,:,27]
            ]
        combined_array = np.stack(arrays_to_concat, axis=2)
        REAL_DATA = torch.tensor(combined_array).float()


        # PROCESS Holdout Data
        processed_X, processed_y, processed_c, time_info = processing.process_data_for_synthesizer(X_test, y_test, c_test, choose_features_for_compatibility)
        time_info_holdout = time_info.float()

        arrays_to_concat = [
            processed_X[:,:,0], # heart rate    
            np.transpose(m_test, (0, 2, 1))[:,:,22], # heart rate mask
            processed_X[:,:,1], # respiration
            np.transpose(m_test, (0, 2, 1))[:,:,42], # respiration mask
            processed_X[:,:,2], # o2 saturation
            np.transpose(m_test, (0, 2, 1))[:,:,35], # o2 saturation mask
            processed_X[:,:,3], # mean arterial pressure
            np.transpose(m_test, (0, 2, 1))[:,:,27]
            ]
        combined_array = np.stack(arrays_to_concat, axis=2)
        HOLDOUT_DATA = torch.tensor(combined_array).float()
        HOLDOUT_COND = torch.concatenate((processed_c, processed_y), axis=2)

        # Process Holdout Validation Data
        processed_X, processed_y, processed_c, time_info = processing.process_data_for_synthesizer(X_val, y_val, c_val, choose_features_for_compatibility)
        time_info_holdout_val = time_info.float()

        arrays_to_concat = [
            processed_X[:,:,0], # heart rate    
            np.transpose(m_val, (0, 2, 1))[:,:,22], # heart rate mask
            processed_X[:,:,1], # respiration
            np.transpose(m_val, (0, 2, 1))[:,:,42], # respiration mask
            processed_X[:,:,2], # o2 saturation
            np.transpose(m_val, (0, 2, 1))[:,:,35], # o2 saturation mask
            processed_X[:,:,3], # mean arterial pressure
            np.transpose(m_val, (0, 2, 1))[:,:,27]
            ]
        combined_array = np.stack(arrays_to_concat, axis=2)
        HOLDOUT_VAL_DATA = torch.tensor(combined_array).float()
        HOLDOUT_VAL_COND = torch.concatenate((processed_c, processed_y), axis=2)




        metadata = {
            'model_version': model_version,
            'genmodel_timestamp': timestamp,
            'important_features_names': ['hr','hr_mask','resp','resp_mask','o2','o2_mask','map','map_mask'],
            'number of features': REAL_DATA.shape[2],
            'seq_len': REAL_DATA.shape[1],
            'seed': seed,
            'patient_length': REAL_DATA.shape[0],
            'numerical_processing': numerical_processing,
            'cond_names': cond_names.tolist() if hasattr(cond_names, 'tolist') else cond_names,
            'size of conditioning tensor': REAL_COND.shape[2]
        }
        metadata.update(data_params)
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
            
        ################################################################################################################
        # Checking Processed Data #
        ################################################################################################################

        print(f"Shape of the response data: {REAL_DATA.shape}")
        print(f"Shape of the condition data: {REAL_COND.shape}")




        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
                                                            # Training #
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        efficient = True
        full_metadata = f'auto_mmd_{auto_mmd_weight}_auto_cons_{auto_consistency_weight}_diff_mmd_{diff_mmd_weight}_diff_cons_{diff_consistency_weight}'
        # metadata = f'{id}'



        ################################################################################################################
        # Defining Model Parameters #
        ################################################################################################################
        if train_models:
            VAE_training = 50000
            diff_training = 50000
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ###### Auto-encoder Parameters ######
            n_epochs = VAE_training; eps = 1e-5
            weight_decay = 1e-6; lr = 2e-4; hidden_size = 200; num_layers = 2; batch_size = 100
            channels = 64; min_beta = 1e-5; max_beta = 0.1; emb_dim = 128; time_dim = time_info_real.shape[2];  lat_dim = REAL_DATA.shape[2]; threshold = 1

            if lat_dim > REAL_DATA.shape[2]:
                raise ValueError("lat_dim should be less than the number of features.")

            ###### Diffusion Parameters ######
            n_epochs = diff_training; hidden_dim = 256; num_layers = 3; diffusion_steps = 1000;


            new_params = {
                "VAE_training": VAE_training,
                "diff_training": diff_training,
                "device": str(device),
                "imputation strategy": "randomly select from imputed patients.", # "drop missing values"
                "eps" : eps,
                "auto_weight_decay" : weight_decay,
                "auto_lr" : lr,
                "auto_hidden_size" : hidden_size,
                "auto_num_layers" : num_layers,
                "auto_batch_size" : batch_size,
                "auto_channels" : channels,
                "auto_min_beta" : min_beta,
                "auto_max_beta" : max_beta,
                "auto_emb_dim" : emb_dim,
                "auto_time_dim" : time_dim,
                "auto_lat_dim" : lat_dim,
                "auto_threshold" : threshold,
                "diff_hidden_dim" : hidden_dim,
                "diffusion_steps" : diffusion_steps,
                "diff_num_layers" : num_layers,
                "auto_mmd_weight" : auto_mmd_weight,
                "auto_consistency_weight" : auto_consistency_weight,
                "diff_mmd_weight" : diff_mmd_weight,
                "diff_consistency_weight" : diff_consistency_weight    
            }   

            # Call the method
            tdf_helper.append_new_params_to_metadata(output_dir, new_params)

            # Path to the metadata JSON file
            metadata_path = os.path.join(output_dir, 'metadata.json')
            # Read the existing JSON file
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Extract the parameters
            patient_length = metadata.get('patient_length')
            imputation_strategy = metadata.get('imputation strategy')
            number_of_features = metadata.get('number of features')




            ################################################################################################################
            # WANDB Initialization #
            ################################################################################################################

            config = dict(
                model = "TimeAutoDiff",
                patient_length = patient_length,
                imputation_strategy = imputation_strategy,
                number_of_features = number_of_features,
                epochs_VAE = VAE_training,
                epochs_diffusion = diff_training,
                pred_task = task_name,
                data_name = data_name,
            )

            use_cuda = torch.cuda.is_available()
            # wandb.init(
            #     project = 'TimeAutoDiff',
            #     config = config,
            #     name = output_dir.split('/')[-1],
            # )

            ################################################################################################################
            # Auto-encoder Training #
            ################################################################################################################
        
        if train_auto:
            torch.cuda.empty_cache()
            if efficient:
                ds = timeautodiff.train_autoencoder(REAL_DATA, channels, hidden_size, num_layers, lr, weight_decay, n_epochs,
                                                            batch_size, min_beta, max_beta, emb_dim, time_dim, lat_dim, device,output_dir, checkpoints=True,
                                                            mmd_weight = auto_mmd_weight, consistency_weight = auto_consistency_weight, use_wandb=False)
            # Save Autoencoder
            ae = ds[0]
            ae.save_model(os.path.join(output_dir, 'autoencoder'))
            # Save latent features
            latent_features = ds[1]
            processing.save_tensor(latent_features,output_dir, 'latent_features.pt')
            print("Latent features saved successfully.")
        else:
            latent_features = torch.load(os.path.join(output_dir, 'latent_features.pt'))
            ae = timeautodiff.DeapStack.load_model(os.path.join(output_dir, 'autoencoder.pt'))
            
        ################################################################################################################
        # Diffusion Training #
        ################################################################################################################
        if train_diff:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            num_classes = len(latent_features)

            new_params = {
                "diff_num_classes" : num_classes,
            }   
            # Call the method
            tdf_helper.append_new_params_to_metadata(output_dir, new_params)

            diff = timeautodiff.train_diffusion(latent_features,
                                                REAL_COND,
                                                time_info_real,
                                                hidden_dim,
                                                num_layers,
                                                diffusion_steps,
                                                n_epochs,output_dir,
                                                checkpoints = True,
                                                num_classes = num_classes,
                                                mmd_weight = diff_mmd_weight,
                                                consistency_weight = diff_consistency_weight,
                                                use_wandb=False)
        with open('output_metadata.txt', 'a') as f:
            f.write(f"Experiment ID: {exp_id}, Metadata: {metadata}, Full Metadata: {full_metadata}, Output Directory: {output_dir}\n")
        models.append({ 'ae': ae, 'diff': diff })
        # Clear cache after each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    ################################################################################################################
    # Evaluating #
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    print('Generating synthetic data...')
    
    synth_data_list = []
    synth_data_y_list = []

    # Generate synthetic data

    conditioning = REAL_COND
    time_info = time_info_real
    for model in models:
        print(f'Generating synthetic data for model {i+1} of 5...')
        _synth_data = tdf_helper.generate_synthetic_data_in_batches(model, conditioning, time_info, 
                                                                        batch_size = 10000)
        
        synth_data_list.append(_synth_data.cpu().numpy())
        synth_data_y_list.append(REAL_COND[:, 0, -1].cpu().numpy().reshape(-1,))

    print('Synthetic data generated successfully.')
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
                                                        # Train dowsnstream Models #
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    ## 

    tstr_gru_params = {
        'input_size': HOLDOUT_DATA.shape[2],  
        'batch_size': 64,
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.1,
        'epochs': 50,
        'eval_step': 10,
        'lr': 0.0005,
        'lr_decay_step': 20,
        'l2': 0.001,
        'multi_metric': True,
        'evaluation': 'gru',
        'eval_early_stop_patience': 20,
        'out_path': f'outputs/{task_name}/{data_name}/TimeAutoDiff/',
        'eval_mode': 'tstr',
        'device': 'cuda',
        'seed': 0
    }

    tstr_metrics = ['auroc', 'auprc', 'sens', 'spec', 'bacc_sens', 'f1_sens', 'tp', 'tn', 'fp', 'fn',
                    'sens_opt', 'spec_opt', 'bacc_opt', 'f1_opt', 'tp_opt', 'tn_opt', 'fp_opt', 'fn_opt']
    trts_metrics = tstr_metrics
    subgroup_results = []

    y_real = REAL_COND[:, 0, -1].cpu().numpy().reshape(-1,)
    y_holdout_train = HOLDOUT_COND[:, 0, -1].cpu().numpy().reshape(-1,)
    y_holdout_val = HOLDOUT_VAL_COND[:, 0, -1].cpu().numpy().reshape(-1,)

    downstream_models = tdf_helper.train_independent_models(HOLDOUT_DATA, HOLDOUT_VAL_DATA, y_holdout_train, y_holdout_val,
                                                                num_models = 5,
                                                                output_dir = tstr_gru_params['out_path'], 
                                                                verbose = True)
    
    
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
                                                        # Evaluate Synthetic Data #
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    results = vis.evaluate_synthetic_data(
                real_data = REAL_DATA.cpu().numpy(),
                synth_data_list = synth_data_list,
                feature_names=  feature_names,
                subset = 'training',
                synth_data_y_list = synth_data_y_list,
                real_data_y = y_real,
                indep_real = HOLDOUT_DATA.cpu().numpy(),
                indep_real_y = y_holdout_train,
                evaluations=[ 'disc', 'pred',  'trts', 'tstr'],
                trts_models=downstream_models,
                trts_metrics = trts_metrics,
                tstr_metrics = tstr_metrics,
                trts_abs = True,
                tstr_model_params=tstr_gru_params,
                tstr_diff_timestamps=diff_timestamp,
                tstr_model_counts = 5,
                disc_counts = 5,
                pred_counts = 5,
                # temp_disc_counts=1,
                # corr_iterations=1,
                intersectional = False,
                )
                
    eval_results = pd.DataFrame(results)
    eval_results = eval_results[eval_results['metric'].isin(['auroc','discriminative','predictive', 'kl_divergence'])]          
    eval_results_dict = eval_results.groupby(['evaluation'])['overall'].agg(['mean','std']).transpose().to_dict()
                        
    eval_path = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0_ecml_conditional_autodiff_generation/scripts_training_and_evaluating_only_consis'      
    # Save results
    try:
        df = pd.read_csv(f'{eval_path}/evaluations_fixed.csv')
    except FileNotFoundError:
        df = pd.DataFrame()

    new_row = {
        'data_name': data_name,
        'task_name': task_name,
        'standardize': standardize,
        'data_timestamp': processed_data_timestamp,
        'model': diff_timestamp,
        'auto_mmd_weight': auto_mmd_weight,
        'auto_consistency_weight': auto_consistency_weight,
        'diff_mmd_weight': diff_mmd_weight,
        'diff_consistency_weight': diff_consistency_weight,
        'exp_id': exp_id,
        'VAE_training': 50000,
        'diff_training': 50000,
        'disc_mean': eval_results_dict['discriminative']['mean'],
        'disc_std': eval_results_dict['discriminative']['std'],
        # 'temp_disc_mean': eval_results['temp_disc']['mean'],
        # 'temp_disc_std': eval_results['temp_disc']['std'],
        # 'corr_mean': eval_results['corr']['mean'],
        # 'corr_std': eval_results['corr']['std'],
        'trts_rpd_mean': eval_results_dict['trts_rpd']['mean'],
        'trts_rpd_std': eval_results_dict['trts_rpd']['std'],
        'trts_real_mean': eval_results_dict['trts_real']['mean'],
        'trts_real_std': eval_results_dict['trts_real']['std'],
        'trts_synth_mean': eval_results_dict['trts_synth']['mean'],
        'trts_synth_std': eval_results_dict['trts_synth']['std'],
        'trts_kl_mean': eval_results_dict['trts_']['mean'],
        'trts_kl_std': eval_results_dict['trts_']['std'],
        'tstr_rpd_mean': eval_results_dict['tstr_rpd']['mean'],
        'tstr_rpd_std': eval_results_dict['tstr_rpd']['std'],
        'tstr_real_mean': eval_results_dict['tstr_real']['mean'],
        'tstr_real_std': eval_results_dict['tstr_real']['std'],
        'tstr_synth_mean': eval_results_dict['tstr_synth']['mean'],
        'tstr_synth_std': eval_results_dict['tstr_synth']['std'],
        'pred_mean': eval_results_dict['predictive']['mean'],
        'pred_std': eval_results_dict['predictive']['std']
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(f'{eval_path}/evaluations_fixed.csv', index=False)    
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TimeAutoDiff model with regularization weights')
    parser.add_argument('--task_name', type=str, required=True, help='task_name')
    parser.add_argument('--data_name', type=str, required=True, help='data_name')
    parser.add_argument('--exp_id', type=str, required=True, help='experiment id')    

    # Add new arguments for weights
    parser.add_argument('--auto_mmd_weight', type=float, default=0.0,
                      help='MMD weight for autoencoder (default: 0.0)')
    parser.add_argument('--auto_consistency_weight', type=float, default=0.0,
                      help='Consistency weight for autoencoder (default: 0.0)')
    parser.add_argument('--diff_mmd_weight', type=float, default=0.1,
                      help='MMD weight for diffusion (default: 0.1)')
    parser.add_argument('--diff_consistency_weight', type=float, default=0.1,
                      help='Consistency weight for diffusion (default: 0.1)')

    args = parser.parse_args()
    
    main(task_name=args.task_name, 
         data_name=args.data_name, 
         exp_id=args.exp_id,
         auto_mmd_weight=args.auto_mmd_weight,
         auto_consistency_weight=args.auto_consistency_weight,
         diff_mmd_weight=args.diff_mmd_weight,
         diff_consistency_weight=args.diff_consistency_weight)
