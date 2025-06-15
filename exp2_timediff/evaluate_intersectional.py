def main(data_name, task_name):

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
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch

    import wandb

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
    processed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/1b_ecml_unconditional_timediff_generation/outputs/{task_name}/{data_name}/processed/'
    intermed_output_path = f'/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/1b_ecml_unconditional_timediff_generation/outputs/{task_name}/{data_name}/intermed/'
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
    REAL_COND = processed_c
    REAL_Y = processed_y
    conds_train = np.stack([y_train, c_train[:,0], c_train[:,1], c_train[:,2]], axis=1)
    static_train = pd.DataFrame(conds_train,columns=['outcome','gender','ethnicity','age_group'])
        
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
    ############# LOAD TIMEDIFF SAMPLES ##############
    ##################################################

    generated_samples_path = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/training_and_evaluating/generated_samples'

    synth_data_list_TIMEDIFF = []
    synth_data_y_list_TIMEDIFF = []
    synth_data_c_list_TIMEDIFF = []
    # Generate synthetic data

    generated_samples_filenames = [file for file in os.listdir(generated_samples_path) if file.startswith(f'timediff_{data_name}_{task_name}')]

    for i,filename in enumerate(generated_samples_filenames):

        print(f'Loading {i+1} of {len(generated_samples_filenames)} samples')

        _synth_data = np.load(f'{generated_samples_path}/{filename}')
        synth_data_list_TIMEDIFF.append(np.transpose(_synth_data[:,:8], (0, 2, 1)))
        synth_data_y_list_TIMEDIFF.append(_synth_data[:,8,0])
        synth_data_c_list_TIMEDIFF.append(_synth_data[:,9:,0])

    print(f'Loaded {len(synth_data_list_TIMEDIFF)} samples')




    ##################################################
    ############# LOAD TIMEAUTODIFF SAMPLES ##############
    ##################################################
    generated_samples_path = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0_ecml_conditional_autodiff_generation/scripts_generating/generated_samples_baseline'

    synth_data_list_TIMEAUTODIFF_baseline = []
    synth_data_y_list_TIMEAUTODIFF_baseline = []
    synth_data_c_list_TIMEAUTODIFF_baseline = []
    # Generate synthetic data

    generated_samples_filenames = [file for file in os.listdir(generated_samples_path) if file.startswith(f'timeautodiff_{data_name}_{task_name}')]

    for i,filename in enumerate(generated_samples_filenames):

        print(f'Loading {i+1} of {len(generated_samples_filenames)} samples')

        _synth_data = np.load(f'{generated_samples_path}/{filename}')
        synth_data_list_TIMEAUTODIFF_baseline.append(_synth_data)
        synth_data_y_list_TIMEAUTODIFF_baseline.append(REAL_Y[:,0,:].cpu().numpy().reshape(-1,))
        synth_data_c_list_TIMEAUTODIFF_baseline.append(REAL_COND[:,0,:].cpu().numpy())

    print(f'Loaded {len(synth_data_list_TIMEAUTODIFF_baseline)} samples')

    ##################################################
    ############# LOAD TIMEAUTODIFF Enhanced SAMPLES ##############
    ##################################################
    generated_samples_path = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0_ecml_conditional_autodiff_generation/scripts_generating/generated_samples_enhanced'

    synth_data_list_TIMEAUTODIFF_enhanced = []
    synth_data_y_list_TIMEAUTODIFF_enhanced = []
    synth_data_c_list_TIMEAUTODIFF_enhanced = []
    # Generate synthetic data

    generated_samples_filenames = [file for file in os.listdir(generated_samples_path) if file.startswith(f'timeautodiff_{data_name}_{task_name}')]

    for i,filename in enumerate(generated_samples_filenames):

        print(f'Loading {i+1} of {len(generated_samples_filenames)} samples')

        _synth_data = np.load(f'{generated_samples_path}/{filename}')
        synth_data_list_TIMEAUTODIFF_enhanced.append(_synth_data)
        synth_data_y_list_TIMEAUTODIFF_enhanced.append(REAL_Y[:,0,:].cpu().numpy().reshape(-1,))
        synth_data_c_list_TIMEAUTODIFF_enhanced.append(REAL_COND[:,0,:].cpu().numpy())

    print(f'Loaded {len(synth_data_list_TIMEAUTODIFF_enhanced)} samples')


        

    ##################################################
    ############# Downstream Model      ##############
    ##################################################

    import timeautodiff.processing_simple as processing
    import timeautodiff.helper_simple as tdf_helper
    import timeautodiff.timeautodiff_v4_efficient_simple as timeautodiff
    import evaluation_framework.vis as vis
    import torch
    import pandas as pd

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

    y_real = REAL_Y[:, 0, -1].cpu().numpy().reshape(-1,)
    y_holdout_train = HOLDOUT_Y[:, 0, -1].cpu().numpy().reshape(-1,)
    y_holdout_val = HOLDOUT_VAL_Y[:, 0, -1].cpu().numpy().reshape(-1,)

    downstream_models = tdf_helper.train_independent_models(HOLDOUT_DATA, HOLDOUT_VAL_DATA, y_holdout_train, y_holdout_val,
                                                                num_models = 5,
                                                                output_dir = tstr_gru_params['out_path'], 
                                                                verbose = True)



    ##################################################
    ### Run Intersectional Analysis ### 
    ##################################################
    static_train['subgroup'] = static_train['outcome'].astype(int).astype(str) + '_' + static_train['gender'].astype(int).astype(str) + '_' + static_train['ethnicity'].astype(int).astype(str) + '_' + static_train['age_group'].astype(int).astype(str) 
    static_train['subgroup_without_label'] =  static_train['gender'].astype(int).astype(str) + '_' + static_train['ethnicity'].astype(int).astype(str) + '_' + static_train['age_group'].astype(int).astype(str) 

    subgroup_counts = pd.DataFrame(static_train['subgroup'].value_counts().reset_index())
    subgroup_counts.columns = ['subgroup', 'count']
    subgroup_counts['oracle_size'] = (0.8*subgroup_counts['count']).round().astype(int)
    subgroup_counts['test_size'] =   (0.2*subgroup_counts['count']).round().astype(int)

    subgroup_counts_without_label = pd.DataFrame(static_train['subgroup_without_label'].value_counts().reset_index())
    subgroup_counts_without_label.columns = ['subgroup', 'count']
    subgroup_counts_without_label['oracle_size'] = (0.8*subgroup_counts_without_label['count']).round().astype(int)
    subgroup_counts_without_label['test_size'] = (0.2*subgroup_counts_without_label['count']).round().astype(int)



    # Create a dictionary to store indices for each subgroup
    all_results = []
    diff_Results = []
    random_states = [42,87,123,13,900]

    # For each subgroup in subgroup_counts
    for _, row in subgroup_counts_without_label.iterrows():
        print('Subgroup: ', row['subgroup'])
        # Split the subgroup code into individual values
        subgroup = row['subgroup']
        oracle_size = row['oracle_size']
        test_size = row['test_size']
        gen, eth, age = map(int, subgroup.split('_'))  # Ignore outcome
        
        # Create boolean mask for this subgroup using only demographics
        mask = (static_train.gender == gen) & \
            (static_train.ethnicity == eth) & \
            (static_train.age_group == age)
        
        # Store the indices where mask is True
        # Get all indices for this subgroup
        target_subgroup_indices = static_train[mask].index.values
        
        # Get outcome label for this subgroup
        subgroup_outcome = static_train.loc[target_subgroup_indices, 'outcome'].values
        unique_classes, class_counts = np.unique(subgroup_outcome, return_counts=True)
        min_class_count = min(class_counts)
        
        # Randomly select N examples for first group
        for seed in random_states:
            try:
                # Check if we have enough samples for the requested oracle and test sizes
                total_samples = len(target_subgroup_indices)
                if total_samples < oracle_size:
                    print(f"Warning: Subgroup {subgroup} has only {total_samples} samples, but oracle_size is {oracle_size}")
                    oracle_size = total_samples
                    test_size = 0
                elif total_samples < oracle_size + test_size:
                    print(f"Warning: Subgroup {subgroup} has only {total_samples} samples, but oracle_size + test_size = {oracle_size + test_size}")
                    test_size = max(0, total_samples - oracle_size)
                
                if min_class_count >= 2 and total_samples > oracle_size:
                    remaining_size = total_samples - oracle_size
                    if remaining_size > 0:
                        # Try stratified split: oracle vs remaining, stratified by outcome
                        oracle_group_indices, remaining_indices = train_test_split(
                            target_subgroup_indices,
                            test_size=remaining_size,
                            stratify=subgroup_outcome,
                            random_state=seed
                        )
                        
                        # Get outcome labels for remaining indices
                        remaining_outcome = static_train.loc[remaining_indices, 'outcome'].values
                        unique_remaining, remaining_counts = np.unique(remaining_outcome, return_counts=True)
                        
                        # Check if we can stratify the test split
                        if len(remaining_indices) >= test_size and test_size > 0 and min(remaining_counts) >= 2:
                            unused_size = len(remaining_indices) - test_size
                            if unused_size > 0:
                                # Stratified split: test vs unused, stratified by outcome
                                test_group_indices, _ = train_test_split(
                                    remaining_indices,
                                    test_size=unused_size,
                                    stratify=remaining_outcome,
                                    random_state=seed
                                )
                            else:
                                # Use all remaining indices as test set
                                test_group_indices = remaining_indices
                        elif test_size > 0:
                            # Random split for test set if stratification not possible
                            np.random.seed(seed)
                            test_group_indices = np.random.choice(remaining_indices, size=min(test_size, len(remaining_indices)), replace=False)
                        else:
                            # No test set needed
                            test_group_indices = np.array([])
                    else:
                        # All samples go to oracle
                        oracle_group_indices = target_subgroup_indices
                        test_group_indices = np.array([])
                else:
                    # Fall back to random splitting for both oracle and test
                    np.random.seed(seed)
                    actual_oracle_size = min(oracle_size, total_samples)
                    oracle_group_indices = np.random.choice(target_subgroup_indices, size=actual_oracle_size, replace=False)
                    remaining_indices = np.setdiff1d(target_subgroup_indices, oracle_group_indices)
                    if test_size > 0 and len(remaining_indices) > 0:
                        test_group_indices = np.random.choice(remaining_indices, size=min(test_size, len(remaining_indices)), replace=False)
                    else:
                        test_group_indices = np.array([])
                    
            except ValueError as e:
                print(f"Stratification failed for subgroup {subgroup}, using random split: {e}")
                # Fall back to random splitting
                np.random.seed(seed)
                oracle_group_indices = np.random.choice(target_subgroup_indices, size=oracle_size, replace=False)
                remaining_indices = np.setdiff1d(target_subgroup_indices, oracle_group_indices)
                test_group_indices = np.random.choice(remaining_indices, size=min(test_size, len(remaining_indices)), replace=False)
            
            # Get X data for real oracle and test sets
            X_subgroup_oracle = REAL_DATA[oracle_group_indices]
            oracle_actual_size = X_subgroup_oracle.shape[0]
            
            # get y data for real oracle and test sets
            y_subgroup_oracle = REAL_Y[oracle_group_indices][:, 0, -1].cpu().numpy().reshape(-1,)
            
            # Handle test set (might be empty)
            if len(test_group_indices) > 0:
                X_subgroup_test = REAL_DATA[test_group_indices]
                y_subgroup_test = REAL_Y[test_group_indices][:, 0, -1].cpu().numpy().reshape(-1,)
                test_actual_size = X_subgroup_test.shape[0]
            else:
                X_subgroup_test = None
                y_subgroup_test = None
                test_actual_size = 0
            
            # === NEW: Random set from all subgroups ===
            # Sample random indices from the entire dataset (all subgroups) with same size as oracle
            np.random.seed(seed)
            all_available_indices = static_train.index.values
            random_set_size = oracle_actual_size
            if len(all_available_indices) >= random_set_size:
                random_group_indices = np.random.choice(
                    all_available_indices, 
                    size=random_set_size, 
                    replace=False
                )
            else:
                random_group_indices = all_available_indices
            
            # Get X and y data for random set
            X_random_set = REAL_DATA[random_group_indices]
            y_random_set = REAL_Y[random_group_indices][:, 0, -1].cpu().numpy().reshape(-1,)
            random_actual_size = X_random_set.shape[0]
            
            # Get synthetic data - now handling different demographic labels
            for i, _ in enumerate(synth_data_list_TIMEAUTODIFF_baseline):
                
                # === TIMEDIFF Data Processing ===
                # Get the corresponding static_train for this TIMEDIFF dataset
                static_train_timediff_i = pd.DataFrame(synth_data_c_list_TIMEDIFF[i], columns=['gender', 'ethnicity', 'age_group'])
                # Create mask for this subgroup in TIMEDIFF demographics
                mask_timediff = (static_train_timediff_i.gender == gen) & \
                            (static_train_timediff_i.ethnicity == eth) & \
                            (static_train_timediff_i.age_group == age)
                
                # Get indices for this subgroup in TIMEDIFF data
                timediff_subgroup_indices = static_train_timediff_i[mask_timediff].index.values
                
                # Sample oracle_size indices from TIMEDIFF subgroup
                np.random.seed(seed)
                if len(timediff_subgroup_indices) >= oracle_size:
                    oracle_group_indices_timediff = np.random.choice(
                        timediff_subgroup_indices, 
                        size=oracle_size, 
                        replace=False
                    )
                else:
                    oracle_group_indices_timediff = timediff_subgroup_indices
                
                # Get TIMEDIFF synthetic data
                _synth_data_TIMEDIFF = synth_data_list_TIMEDIFF[i][oracle_group_indices_timediff]
                X_synth_TIMEDIFF = _synth_data_TIMEDIFF
                y_synth_TIMEDIFF = synth_data_y_list_TIMEDIFF[i][oracle_group_indices_timediff]
                
                # === TIMEAUTODIFF Data Processing ===
                # Get the corresponding static_train for this TIMEAUTODIFF dataset
                static_train_timeautodiff_i = pd.DataFrame(synth_data_c_list_TIMEAUTODIFF_baseline[i], columns=['gender', 'ethnicity', 'age_group'])
                
                # Create mask for this subgroup in TIMEAUTODIFF demographics
                mask_timeautodiff = (static_train_timeautodiff_i.gender == gen) & \
                                (static_train_timeautodiff_i.ethnicity == eth) & \
                                (static_train_timeautodiff_i.age_group == age)
                
                # Get indices for this subgroup in TIMEAUTODIFF data
                timeautodiff_subgroup_indices = static_train_timeautodiff_i[mask_timeautodiff].index.values
                
                # Sample oracle_size indices from TIMEAUTODIFF subgroup
                np.random.seed(seed)
                if len(timeautodiff_subgroup_indices) >= oracle_size:
                    oracle_group_indices_timeautodiff = np.random.choice(
                        timeautodiff_subgroup_indices, 
                        size=oracle_size, 
                        replace=False
                    )
                else:
                    oracle_group_indices_timeautodiff = timeautodiff_subgroup_indices
                
                # Get TIMEAUTODIFF synthetic data
                _synth_data_TIMEAUTODIFF_baseline = synth_data_list_TIMEAUTODIFF_baseline[i][oracle_group_indices_timeautodiff]
                X_synth_TIMEAUTODIFF_baseline = _synth_data_TIMEAUTODIFF_baseline
                y_synth_TIMEAUTODIFF_baseline = synth_data_y_list_TIMEAUTODIFF_baseline[i][oracle_group_indices_timeautodiff]
                
                
                _synth_data_TIMEAUTODIFF_enhanced = synth_data_list_TIMEAUTODIFF_enhanced[i][oracle_group_indices_timeautodiff]
                X_synth_TIMEAUTODIFF_enhanced = _synth_data_TIMEAUTODIFF_enhanced
                y_synth_TIMEAUTODIFF_enhanced = synth_data_y_list_TIMEAUTODIFF_enhanced[i][oracle_group_indices_timeautodiff]
                # === Model Evaluation ===
                for model_index, downstream_model in enumerate(downstream_models):
                    results_synth_TIMEDIFF = downstream_model.evaluate(X_synth_TIMEDIFF, y_synth_TIMEDIFF)
                    results_synth_TIMEAUTODIFF_baseline = downstream_model.evaluate(X_synth_TIMEAUTODIFF_baseline, y_synth_TIMEAUTODIFF_baseline)
                    results_synth_TIMEAUTODIFF_enhanced = downstream_model.evaluate(X_synth_TIMEAUTODIFF_enhanced, y_synth_TIMEAUTODIFF_enhanced)
                    results_oracle = downstream_model.evaluate(X_subgroup_oracle, y_subgroup_oracle)
                    
                    # Evaluate on test set (if available)
                    if X_subgroup_test is not None:
                        results_test = downstream_model.evaluate(X_subgroup_test, y_subgroup_test)
                    else:
                        results_test = None
                    
                    # Evaluate on random set from all subgroups
                    results_random = downstream_model.evaluate(X_random_set, y_random_set)
                    
                    # Store all results
                    all_results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'oracle',
                        'evaluation_size': oracle_actual_size,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        **results_oracle
                    })
                    
                    # Add test results only if test set exists
                    if results_test is not None:
                        all_results.append({
                            'subgroup': subgroup,
                            'evaluated_on': 'test',
                            'evaluation_size': test_actual_size,
                            'subgroup_size': test_actual_size,
                            'test_oracle_random_state': seed,
                            'synth_data_index': i,
                            'model': model_index,
                            **results_test
                        })
                    
                    # Add random set results
                    all_results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'random_all_subgroups',
                        'evaluation_size': random_actual_size,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        **results_random
                    })
                    
                    all_results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'synthetic_timediff',
                        'evaluation_size': X_synth_TIMEDIFF.shape[0] if X_synth_TIMEDIFF is not None else None,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        **results_synth_TIMEDIFF
                    })
                    all_results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'synthetic_timeautodiff',
                        'evaluation_size': X_synth_TIMEAUTODIFF_baseline.shape[0] if X_synth_TIMEAUTODIFF_baseline is not None else None,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        **results_synth_TIMEAUTODIFF_baseline
                    })
                    all_results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'synthetic_timeautodiff_enhanced',
                        'evaluation_size': X_synth_TIMEAUTODIFF_enhanced.shape[0] if X_synth_TIMEAUTODIFF_enhanced is not None else None,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        **results_synth_TIMEAUTODIFF_enhanced
                    })

                    # Store difference results
                    diff_Results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'synthetic_timediff',
                        'evaluation_size': X_synth_TIMEDIFF.shape[0] if X_synth_TIMEDIFF is not None else None,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        'diff': np.abs(results_synth_TIMEDIFF['auroc'] - results_oracle['auroc']) if results_synth_TIMEDIFF.get('auroc', None) is not None and results_oracle['auroc'] is not None else None
                    })
                    diff_Results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'synthetic_timeautodiff',
                        'evaluation_size': X_synth_TIMEAUTODIFF_baseline.shape[0] if X_synth_TIMEAUTODIFF_baseline is not None else None,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        'diff': np.abs(results_synth_TIMEAUTODIFF_baseline['auroc'] - results_oracle['auroc']) if results_synth_TIMEAUTODIFF_baseline.get('auroc', None) is not None and results_oracle['auroc'] is not None else None
                    })
                    diff_Results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'synthetic_timeautodiff_enhanced',
                        'evaluation_size': X_synth_TIMEAUTODIFF_enhanced.shape[0] if X_synth_TIMEAUTODIFF_enhanced is not None else None,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        'diff': np.abs(results_synth_TIMEAUTODIFF_enhanced['auroc'] - results_oracle['auroc']) if results_synth_TIMEAUTODIFF_enhanced.get('auroc', None) is not None and results_oracle['auroc'] is not None else None
                    })
                    
                    # Add test difference (if test set exists)
                    if results_test is not None:
                        diff_Results.append({
                            'subgroup': subgroup,
                            'evaluated_on': 'test',
                            'evaluation_size': test_actual_size,
                            'subgroup_size': test_actual_size,
                            'test_oracle_random_state': seed,
                            'synth_data_index': i,
                            'model': model_index,
                            'diff': np.abs(results_test['auroc'] - results_oracle['auroc']) if results_test['auroc'] is not None and results_oracle['auroc'] is not None else None
                        })
                    
                    # Add random set difference
                    diff_Results.append({
                        'subgroup': subgroup,
                        'evaluated_on': 'random_all_subgroups',
                        'evaluation_size': random_actual_size,
                        'subgroup_size': test_actual_size,
                        'test_oracle_random_state': seed,
                        'synth_data_index': i,
                        'model': model_index,
                        'diff': np.abs(results_random['auroc'] - results_oracle['auroc']) if results_random['auroc'] is not None and results_oracle['auroc'] is not None else None
                    })

    SCRIPTS_PATH = '/home/company/user/projects/AI4Health/notebooks/user/Generative-Models/ehr/4_timediff/icu-autodiff/0c_ecml_conditional_timediff_generation/scripts_evaluating_intersectional/results'

    pd.DataFrame(diff_Results).to_csv(f'{SCRIPTS_PATH}/diff_results_{data_name}_{task_name}.csv', index=False)
    pd.DataFrame(all_results).to_csv(f'{SCRIPTS_PATH}/all_results_{data_name}_{task_name}.csv', index=False)
    
    print(f'Saved results to {SCRIPTS_PATH}/diff_results_{data_name}_{task_name}.csv and {SCRIPTS_PATH}/all_results_{data_name}_{task_name}.csv')
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Intersectional Analysis')
    parser.add_argument('--task_name', type=str, required=True, help='task_name')
    parser.add_argument('--data_name', type=str, required=True, help='data_name')


    args = parser.parse_args()
    
    main(task_name=args.task_name, 
         data_name=args.data_name)

