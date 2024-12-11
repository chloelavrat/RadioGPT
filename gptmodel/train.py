from colorama import Fore, Style
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from core.dataset import AlpacaDataset, SQuADDataset, OpenWebTextDataset
from core.model import GPTlite
from core.utils import train, evaluate

# Additional imports for checkpointing and logging
import time
import numpy as np

# Hyperparameters
block_size = 64
n_embd = 512
n_head = 8
n_layer = 10
dropout = 0.2
learning_rate = 1e-4
max_iters = 50000
batch_size = 16
grad_clip = 0.5
warmup_iters = 1000
checkpoint_interval = 5000  # Save model every 5000 iterations
early_stopping_patience = 5  # Early stopping patience

torch.cuda.empty_cache()

# Load dataset
# squad = SQuADDataset("dataset/SQuAD-train-v2.0.json", block_size)
# alpaca = AlpacaDataset("dataset/alpaca_data_en.json", block_size)
alpaca = AlpacaDataset(
    "dataset/Acquiesce_data_110k_instructions.json", block_size)

# openwebtext = OpenWebTextDataset(block_size)

dataset = alpaca

# Initialize model
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)


model = GPTlite({
    'context_size': 64,  # or whatever value is appropriate
    'vocab_size': dataset.vocab_size,
    'embedding_dim': n_embd,
    'num_heads': n_head,
    'num_layers': n_layer,  # Ensure you have this variable defined
    'dropout': dropout  # Ensure you have this variable defined
}).to(device)

total_params = sum(p.numel() for p in model.parameters())

# Initialize optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                        weight_decay=0.1, betas=(0.9, 0.999))

# Learning rate scheduler
lr_scheduler = OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    epochs=max_iters,
    steps_per_epoch=1,
    pct_start=0.1,  # Warm-up for 10% of training
    div_factor=25,  # Initial lr = max_lr/25
    final_div_factor=1e4  # Final lr = max_lr/10000
)

# print information
print(Fore.GREEN + Style.BRIGHT +
      "Training Conversational GPT Lite model" + Style.RESET_ALL)
print(Fore.CYAN + f"using device: \t\t{device}" + Style.RESET_ALL)
print(Fore.CYAN +
      f"total parameters: \t{total_params / 1e6:.1f}M" + Style.RESET_ALL)
print(Fore.CYAN + f"vocab size: \t\t{dataset.vocab_size}" + Style.RESET_ALL)
print(Fore.CYAN + f"block size: \t\t{block_size}" + Style.RESET_ALL)
print(Fore.CYAN + f"batch size: \t\t{batch_size}" + Style.RESET_ALL)

# Load best model weights if they exist
if os.path.exists("models/best_model.pth"):
    print("-----")
    print(Fore.GREEN + "Loading best model weights from: models/best_model.pth" + Style.RESET_ALL)
    model.load_state_dict(torch.load(
        "models/best_model.pth", map_location=device, weights_only=True))
print("-----")
print(Fore.GREEN + Style.BRIGHT +
      "Training... Let's make some fried eggs!" + Style.RESET_ALL)
print("-----")

# Initialize variables for early stopping
best_val_loss = float('inf')
patience_counter = 0

# Training loop with checkpointing and early stopping
for iter in range(max_iters):
    # Train the model
    train(model, dataset, optimizer, max_iters, batch_size, device,
          grad_clip=grad_clip, warmup_iters=warmup_iters, lr_scheduler=lr_scheduler,
          learning_rate=learning_rate)

    # Checkpointing
    if iter % checkpoint_interval == 0:
        torch.save(model.state_dict(), f"models/checkpoint_iter_{iter}.pth")
        print(Fore.YELLOW +
              f"Checkpoint saved at iteration {iter}" + Style.RESET_ALL)

    # Early stopping logic (assuming you have a validation loss to monitor)
    # Assuming evaluate returns validation loss
    val_loss = evaluate(model, dataset, device)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), "models/best_model.pth")
        print(Fore.GREEN + "Best model saved!" + Style.RESET_ALL)
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(Fore.RED + "Early stopping triggered!" + Style.RESET_ALL)
            break

# Evaluate the model
evaluate(model, dataset, device)
