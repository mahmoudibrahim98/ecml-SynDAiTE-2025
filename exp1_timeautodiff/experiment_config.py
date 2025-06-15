EXPERIMENTS = {
    "exp1_baseline": {
        "name": "No Regularization",
        "weights": {
            "auto_mmd_weight": 0,
            "auto_consistency_weight": 0,
            "diff_mmd_weight": 0,
            "diff_consistency_weight": 0
        }
    },
    
    "exp2a_auto_mmd": {
        "name": "Autoencoder MMD Only",
        "weights": {
            "auto_mmd_weight": 0.1,
            "auto_consistency_weight": 0,
            "diff_mmd_weight": 0,
            "diff_consistency_weight": 0
        }
    },
    
    "exp2b_auto_consistency": {
        "name": "Autoencoder Consistency Only",
        "weights": {
            "auto_mmd_weight": 0,
            "auto_consistency_weight": 0.1,
            "diff_mmd_weight": 0,
            "diff_consistency_weight": 0
        }
    },
    
    "exp2c_auto_both": {
        "name": "Autoencoder Both",
        "weights": {
            "auto_mmd_weight": 0.1,
            "auto_consistency_weight": 0.1,
            "diff_mmd_weight": 0,
            "diff_consistency_weight": 0
        }
    },
    
    "exp3a_diff_mmd": {
        "name": "Diffusion MMD Only",
        "weights": {
            "auto_mmd_weight": 0,
            "auto_consistency_weight": 0,
            "diff_mmd_weight": 0.1,
            "diff_consistency_weight": 0
        }
    },
    
    "exp3b_diff_consistency": {
        "name": "Diffusion Consistency Only",
        "weights": {
            "auto_mmd_weight": 0,
            "auto_consistency_weight": 0,
            "diff_mmd_weight": 0,
            "diff_consistency_weight": 0.1
        }
    },
    
    "exp3c_diff_both": {
        "name": "Diffusion Both",
        "weights": {
            "auto_mmd_weight": 0,
            "auto_consistency_weight": 0,
            "diff_mmd_weight": 0.1,
            "diff_consistency_weight": 0.1
        }
    },
    
    "exp4a_full_balanced": {
        "name": "Full Regularization Balanced",
        "weights": {
            "auto_mmd_weight": 0.1,
            "auto_consistency_weight": 0.1,
            "diff_mmd_weight": 0.1,
            "diff_consistency_weight": 0.1
        }
    },
    
    "exp4b_full_auto_strong": {
        "name": "Full Regularization Strong Auto",
        "weights": {
            "auto_mmd_weight": 0.5,
            "auto_consistency_weight": 0.5,
            "diff_mmd_weight": 0.1,
            "diff_consistency_weight": 0.1
        }
    },
    
    "exp4c_full_diff_strong": {
        "name": "Full Regularization Strong Diff",
        "weights": {
            "auto_mmd_weight": 0.1,
            "auto_consistency_weight": 0.1,
            "diff_mmd_weight": 0.5,
            "diff_consistency_weight": 0.5
        }
    }
} 