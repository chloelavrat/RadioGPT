from colorama import Fore, Style
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from core.dataset import AlpacaDataset
from core.model import GPTlite
from core.utils import download_dataset
from tqdm import tqdm
from config import config

# clear cache
torch.cuda.empty_cache()

# Download dataset if not already downloaded
if not os.path.exists(config["dataset"]["file_path"]):
    download_dataset(config["dataset"]["download_url"],
                     config["dataset"]["file_path"])

# Load dataset
full_dataset = AlpacaDataset(
    config["dataset"]["file_path"], config["model"]["block_size"])

# Split dataset into train and validation
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config["Training"]["batch_size"],
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["Training"]["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Set device
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

# Initialize model
model = GPTlite({
    'context_size': config["model"]["block_size"],
    'vocab_size': full_dataset.vocab_size,
    'embedding_dim': config["model"]["n_embd"],
    'num_heads': config["model"]["n_head"],
    'num_layers': config["model"]["n_layer"],
    'dropout': config["model"]["dropout"]
}).to(device)

# Initialize optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=config["Training"]["learning_rate"],
                        weight_decay=0.1, betas=(0.9, 0.999))

# Learning rate scheduler
lr_scheduler = OneCycleLR(
    optimizer,
    max_lr=config["Training"]["learning_rate"],
    epochs=config["Training"]["epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    div_factor=25,
    final_div_factor=1e4
)

# print information
total_params = sum(p.numel() for p in model.parameters())
print("Training Conversational GPT Lite model")
print(f"using device: \t\t{device}")
print(f"total parameters: \t{total_params / 1e6:.1f}M")
print(f"vocab size: \t\t{full_dataset.vocab_size}")
print(f"block size: \t\t{config['model']['block_size']}")
print(f"batch size: \t\t{config['Training']['batch_size']}")
print(f"train dataset size: \t{len(train_dataset)}")
print(f"validation dataset size: \t{len(val_dataset)}")

# Load best model weights if they exist
if os.path.exists(config["Training"]["save_dir"] + "/best_model.pth"):
    print("Loading best model weights...")
    model.load_state_dict(torch.load(
        config["Training"]["save_dir"] + "/best_model.pth", 
        map_location=device,
        weights_only=True
    ))

if not os.path.exists(config["Training"]["save_dir"]):
    os.makedirs(config["Training"]["save_dir"])

# Initialize variables for early stopping
best_validation_loss = float('inf')
no_improvement_count = 0

# Initialize scaler for mixed precision only if CUDA is available
use_amp = device.type == 'cuda'
scaler = torch.cuda.amp.GradScaler() if use_amp else None


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


for epoch in range(config["Training"]["epochs"]):
    model.train()
    total_train_loss = 0
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (x, y) in enumerate(train_progress_bar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
            if config["Training"]["grad_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["Training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training for CPU/MPS
            logits, loss = model(x, y)
            loss.backward()
            if config["Training"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["Training"]["grad_clip"])
            optimizer.step()

        lr_scheduler.step()
        total_train_loss += loss.item()

        train_progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    # Validation phase
    if (epoch + 1) % config["Training"]["eval_every"] == 0:
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)

        print("-" * 30)
        print(f"Epoch {epoch+1}/{config['Training']['epochs']}")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 30)

        # Save best model
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(),
                       f"{config['Training']['save_dir']}/best_model.pth")
            print(f"New best model saved! Loss: {best_validation_loss:.4f}")
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= config["Training"]["patience"]:
            print(
                f"No improvement in {config['Training']['patience']} evaluations. Early stopping.")
            break

        model.train()

print("Training completed!")
