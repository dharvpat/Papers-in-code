class Config:
    # Model Hyperparameters
    d_model = 512
    nhead = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1

    # Training Parameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    warmup_steps = 4000

    # Paths
    data_path = './data'
    model_save_path = './models/model.pth'
    tokenizer_path = './tokenizer/tokenizer.json'