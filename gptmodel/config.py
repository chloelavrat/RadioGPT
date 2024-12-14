config = {
    "model": {
        "block_size": 64,
        "n_embd": 512,
        "n_head": 8,
        "n_layer": 10,
        "dropout": 0.2
    },
    "Training": {
        "learning_rate": 1e-4,
        "epochs": 50000,
        "eval_every": 1000,
        "batch_size": 16,
        "grad_clip": 0.5,
        "warmup_iters": 1000,
        "patience": 10,
        "save_dir": "models",
        "train_split": 0.8,
    },
    "dataset": {
        "file_path": "dataset/Acquiesce_data_110k_instructions.json",
        "download_url": "https://openfileserver.chloelavrat.com/workshops/RadioGPT/dataset/Acquiesce_data_110k_instructions.json"
    }
}
