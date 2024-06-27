model_cfg = {
    "majajhong_demo": {
        "models": {
            "vit": {"features": [12], "layers": None},
            "expansion": {"features": [3], "layers": 5},
            "fully_connected": {"features": [3], "layers": 5},
        },
        "regions": ["IT"],
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 10
    }
}

analysis_cfg = {
    "majajhong_demo": {
        "analysis": {
            "activation_function": {"features": [3], "layers": 5},
            "local_connectivity": {"features": [3], "layers": 5},
            "layer_1_filters": {"features": [3], "layers": 5},
            "pca": {"features": [3], "layers": 5},
            "non_linearities": {"features": [3], "layers": 5, "variations": ["relu", "gelu", "elu", "abs", "leaky_relu"]},
            "init_types": {"features": [3], "layers": 5, 
                           "variations": ["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "orthogonal"]}
        },
        "regions": "IT",
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 10
    },
    "majajhong_demo_shuffled": {
        "models": {
            "expansion": {"features": [3], "layers": 5}
        },
        "regions": "IT",
        "subjects": ["Tito", "Chabo"],
        "test_data_size": 10
    },
    "places_val_demo": {
        "models": {
            "alexnet": {"features": None, "layers": 5},
            "expansion": {"features": 3, "layers": 5},
        },
    },
    "places_train_demo": {
        "models": {
            "alexnet": {"features": None, "layers": 5},
            "expansion": {"features": 3, "layers": 5},
        },
    }
}
