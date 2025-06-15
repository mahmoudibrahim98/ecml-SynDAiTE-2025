# ...

def main(data_name, task_name, generate_mode):

    import os
    import sys

    os.chdir('/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/TimeDiff/')

    dir = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/0_Synthetic-data-experiments'
    sys.path.append(dir)
    dir2 = '/mnthpc/netapp01/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/TimeDiff/'
    sys.path.append(dir2)
    dir = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff'
    sys.path.append(dir)
    dir = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/0_Synthetic-data-experiments'
    sys.path.append(dir)
    dir2 = '/mnthpc/netapp01/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/TimeDiff/'
    sys.path.append(dir2)


    print(os.getcwd())
    print(os.listdir())
    import argparse
    import time
    # import os
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    from sklearn.model_selection import train_test_split

    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    import wandb
    from helpers.utils import (create_id, is_file_not_on_disk, is_file_on_disk,
                            is_positive_float, is_positive_integer,
                            seed_everything, exists, reverse_normalize)
    from models.ETDiff.gaussian_diffusion import GaussianDiffusion
    from models.ETDiff.mixed_diffusion import MixedDiffusion
    from models.ETDiff.et_diff import ETDiff
    from models.ETDiff.blocks import NeuralCDE, RNN, EncoderDecoderRNN
    from models.ETDiff.custom_utils import TimeSeriesDataset,replace_nan_with_mean, normalize_data
    import numpy as np



    import data_access.ricu_loader as ricu_loader


    # splitting parameters
    train_fraction = 0.2
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
    processed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/outputs/{task_name}/{data_name}/processed/'
    intermed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/outputs/{task_name}/{data_name}/intermed/'
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

    arrays_to_concat = [
        X_train[:, 22:23, :],  # Heart rate values
        m_train[:, 22:23, :],  # Heart rate masks
        X_train[:, 42:43, :],  # Respiration values
        m_train[:, 42:43, :],  # Respiration masks
        X_train[:, 35:36, :],  # O2 saturation values
        m_train[:, 35:36, :],  # O2 saturation masks
        X_train[:, 27:28, :],  # Mean arterial pressure values
        m_train[:, 27:28, :],   # Mean arterial pressure masks,
        np.repeat(y_train.reshape(-1, 1)[:, :, np.newaxis], 25, axis=2), # mortality,
        np.repeat(c_train[:,0].reshape(-1, 1)[:, :, np.newaxis], 25, axis=2), # gender
        np.repeat(c_train[:,1].reshape(-1, 1)[:, :, np.newaxis], 25, axis=2), # ethnicity
        np.repeat(c_train[:,2].reshape(-1, 1)[:, :, np.newaxis], 25, axis=2), # age group
        
    ]
    combined_array = np.concatenate(arrays_to_concat, axis=1)
    print(sum(y_train)/len(y_train))
    print(sum(combined_array[:,-1,0]) / len(combined_array))
    import torch
    raw_data_path = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/training_and_evaluating/raw_data_timediff'
    torch.save(torch.asarray(combined_array), f'{raw_data_path}/{task_name}_{data_name}_outcome_demographic.pt')

    COLUMNS_DICT = {
        "mimiciv": {"numerical": [0, 2, 4, 6, 8], "categorical": [1, 3, 5, 7, 9, 10], "categorical_num_classes": [2, 2, 2, 2, 2, 2]},
        "mimiciii": {"numerical": [0, 2, 4, 6, 8, 10, 12], "categorical": [1, 3, 5, 7, 9, 11, 13, 14], "categorical_num_classes": [2, 2, 2, 2, 2, 2, 2, 2]},
        "eicu": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8,9,10,11], "categorical_num_classes": [2, 2, 2, 2, 2,2,4,4]},
        "mimic": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8,9,10,11], "categorical_num_classes": [2, 2, 2, 2, 2,2,4,4]},
        "hirid": {"numerical": [0, 1, 2, 3, 4, 5, 6], "categorical": [7], "categorical_num_classes": [2]},
        
        # "eicu_with_cond_mor": {"numerical": [0, 2, 4, 6], "categorical": [1, 3, 5, 7, 8,9,10,11], "categorical_num_classes": [2, 2, 2, 2, 2,2,4,4]},

    }
    metadata = f'{task_name}_{data_name}_outcome_demographic'
    load_path = f'{raw_data_path}/{metadata}.pt'
    cut_length = 272
    processed_data_path = f"/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/training_and_evaluating/processed_data_timediff/"

    dataset = TimeSeriesDataset(
        categorical_cols = COLUMNS_DICT[data_name]["categorical"] if data_name in ["eicu", 'mimic'] else None,             # indicate columns that are not time series values (missing indicators)
        data_name = data_name, 
        task_name = task_name,
        load_path = load_path,
        cut_length = cut_length,
        processed_data_path = processed_data_path
    )


    total_numerical = len(COLUMNS_DICT[data_name]["numerical"])
    total_categorical = sum(COLUMNS_DICT[data_name]["categorical_num_classes"])
    total_channels = total_numerical + total_categorical

    model = RNN(
        input_channels = total_channels,
        hidden_channels = total_channels * 4 if data_name in ["stock", "energy", "mimiciv", "mimiciii", "hirid"] else 256,
        output_channels = total_channels,
        layers = 3,
        model = 'lstm',
        dropout = 0,
        bidirectional = True,
        self_condition = False,
        embed_dim = 64,
        time_dim = 256,
    )
    diffusion = MixedDiffusion(
        model = model,
        channels = total_channels,
        seq_length = dataset.seq_length,
        timesteps = 1000,
        auto_normalize = True,
        numerical_features_indices = COLUMNS_DICT[data_name]["numerical"],
        categorical_features_indices = COLUMNS_DICT[data_name]["categorical"],
        categorical_num_classes = COLUMNS_DICT[data_name]["categorical_num_classes"],
        loss_lambda = 0.8,
    )
    run_id = create_id()
    # wandb.init( project = 'ICU-AutoDiff',  name=f"ETDiff_{run_id}")
    check_point_path = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/training_and_evaluating/trained_models_checkpoint'
    os.makedirs(check_point_path, exist_ok=True)
    etdiff = ETDiff(
        diffusion_model = diffusion,
        dataset = dataset,
        sample_every = 10,
        train_batch_size = 32,
        diff_lr = 8e-5,
        diff_num_steps =  700000,         # total training steps 700000 
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,
        amp = False,
        wandb = None,
        check_point_path = f"{check_point_path}/timediff_{metadata}_{time.strftime('%Y%m%d_%H%M%S')}.pt",
        run_id = run_id,
    )

    if generate_mode == 'train':
        etdiff.train()
        print(f"Completed Training ETDiff model for {task_name} on {data_name} with run_id {run_id} for etdiff.")
    elif generate_mode == 'evaluate':
        trained_model_filename = [file for file in os.listdir(check_point_path) if file.startswith(f'timediff_{metadata}')][0]
        trained_model_path = f"{check_point_path}/{trained_model_filename}"
        etdiff.load(trained_model_path)
        print('Loaded trained model',trained_model_filename)
    # metadata = f'{task_name}_{data_name}_outcome_demographic'
    # trained_model_filename = [file for file in os.listdir(check_point_path) if file.startswith(f'timediff_{metadata}')][0]
    # trained_model_path = f"{check_point_path}/{trained_model_filename}"
    # etdiff.load(trained_model_path)
    # print('Loaded trained model',trained_model_filename)
    # sanity_synth = etdiff.ema.ema_model.sample(100)
    # print(sanity_synth.shape)





    ##################################################
    ############# Generate the samples ##############
    ##################################################

    import timeautodiff.processing_simple as processing

    choose_features_for_compatibility = [22,42,35,27]
    # prorcess data for training of generators
    processed_X, processed_y, processed_c, time_info = processing.process_data_for_synthesizer(X_train, y_train, c_train, choose_features_for_compatibility)
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
    REAL_Y = processed_y
    REAL_CONDS = np.stack([y_train, c_train[:,0], c_train[:,1], c_train[:,2]], axis=1)
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
    HOLDOUT_Y = processed_y

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
    HOLDOUT_VAL_Y = processed_y




    ##################################################
    ############# Generate the samples ##############
    ##################################################
    import pandas as pd
    static = pd.DataFrame(REAL_CONDS,columns=['outcome','gender','ethnicity','age_group'])
    static['subgroup'] = static['outcome'].astype(int).astype(str) + '_' + static['gender'].astype(int).astype(str) + '_' + static['ethnicity'].astype(int).astype(str) + '_' + static['age_group'].astype(int).astype(str) 
    static['subgroup_without_label'] =  static['gender'].astype(int).astype(str) + '_' + static['ethnicity'].astype(int).astype(str) + '_' + static['age_group'].astype(int).astype(str) 

    subgroup_counts = pd.DataFrame(static['subgroup'].value_counts().reset_index())
    subgroup_counts.columns = ['subgroup', 'count']
    subgroup_counts['oracle_size'] = (0.8*subgroup_counts['count']).round().astype(int)
    subgroup_counts['test_size'] = (0.2*subgroup_counts['oracle_size']).round().astype(int)


    # Create target combinations from subgroup_counts
    target_combinations = []
    for _, row in subgroup_counts.iterrows():
        mor, gen, eth, age = row['subgroup'].split('_')
        labels = {
            8: int(mor),    # mortality
            9: int(gen),    # gender
            10: int(eth),   # ethnicity
            11: int(age)    # age_group
        }
        target_combinations.append((labels, row['count']))

    import sys
    sys.path.append('/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/evaluating_scripts')
    from sample import SampleCollector
    collector = SampleCollector(etdiff.ema.ema_model)

    # Define target combinations: (conditions, relative_count)


    # Generate multiple datasets
    datasets_params = [
        {
            'num_samples': len(dataset.data),
            'target_combinations': target_combinations,
            'batch_size': 10000
        },
        {
            'num_samples': len(dataset.data),
            'target_combinations': target_combinations,
            'batch_size': 10000
        },
        {
            'num_samples': len(dataset.data),
            'target_combinations': target_combinations,
            'batch_size': 10000
        },
        {
            'num_samples': len(dataset.data),
            'target_combinations': target_combinations,
            'batch_size': 10000
        },
        {
            'num_samples': len(dataset.data),
            'target_combinations': target_combinations,
            'batch_size': 10000
        }
    ]

    results = collector.generate_k_datasets(datasets_params)

    for i, (samples, counts) in enumerate(results):
        print(f"\nDataset {i+1}: {len(samples)} samples")
        print(f"Combination counts: {counts}")
        
    print(f"\nLeftover stats: {collector.get_leftover_stats()}")

    generated_samples_path = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/training_and_evaluating/generated_samples/'


    # You can now access individual datasets:
    # dataset1 = collected_datasets[0]
    # dataset2 = collected_datasets[1]
    for _synth_data, _ in results:
        save_path = f'{generated_samples_path}/timediff_{data_name}_{task_name}_outcome_demographic_samples_{time.strftime("%Y%m%d_%H%M%S")}.npy'
        np.save(save_path, _synth_data)
        time.sleep(5)

        

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Intersectional Analysis')
    parser.add_argument('--task_name', type=str, required=True, help='task_name')
    parser.add_argument('--data_name', type=str, required=True, help='data_name')
    parser.add_argument('--generate_mode', type=str, required=True, help='generate_mode')

    args = parser.parse_args()
    
    main(task_name=args.task_name, 
         data_name=args.data_name,
         generate_mode=args.generate_mode)
