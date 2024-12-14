import os
import torch
from tqdm import tqdm
from config import config

import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

from core.dataset import AlpacaDataset
from core.model import GPTlite
from core.utils import download_dataset


def setup_dataset():
    if not os.path.exists(config["dataset"]["file_path"]):
        download_dataset(config["dataset"]["download_url"],
                         config["dataset"]["file_path"])

    full_dataset = AlpacaDataset(
        config["dataset"]["file_path"], config["model"]["block_size"])

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    return full_dataset, train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset):
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

    return train_loader, val_loader


def setup_training(full_dataset):
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )

    model = GPTlite({
        'context_size': config["model"]["block_size"],
        'vocab_size': full_dataset.vocab_size,
        'embedding_dim': config["model"]["n_embd"],
        'num_heads': config["model"]["n_head"],
        'num_layers': config["model"]["n_layer"],
        'dropout': config["model"]["dropout"]
    }).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["Training"]["learning_rate"],
        weight_decay=0.1,
        betas=(0.9, 0.999)
    )

    return device, model, optimizer


def setup_lr_scheduler(optimizer, train_loader):
    return OneCycleLR(
        optimizer,
        max_lr=config["Training"]["learning_rate"],
        epochs=config["Training"]["epochs"],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1e4
    )


def print_training_info(device, model, full_dataset, train_dataset, val_dataset):
    total_params = sum(p.numel() for p in model.parameters())
    print("Training Conversational GPT Lite model")
    print(f"using device: \t\t{device}")
    print(f"total parameters: \t{total_params / 1e6:.1f}M")
    print(f"vocab size: \t\t{full_dataset.vocab_size}")
    print(f"block size: \t\t{config['model']['block_size']}")
    print(f"batch size: \t\t{config['Training']['batch_size']}")
    print(f"train dataset size: \t{len(train_dataset)}")
    print(f"validation dataset size: \t{len(val_dataset)}")


def load_best_model(model, device):
    model_path = config["Training"]["save_dir"] + "/best_model.pth"
    if os.path.exists(model_path):
        print("Loading best model weights...")
        model.load_state_dict(torch.load(
            model_path,
            map_location=device,
            weights_only=True
        ))


def evaluate(model, dataloader, device, use_amp):
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


def train_epoch(model, train_loader, optimizer, lr_scheduler, device, use_amp, scaler, epoch):
    model.train()
    total_train_loss = 0
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for x, y in train_progress_bar:
        x, y = x.to(device), y.to(device)
        loss = train_step(model, x, y, optimizer, use_amp, scaler)

        lr_scheduler.step()
        total_train_loss += loss.item()

        train_progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return total_train_loss / len(train_loader)


def train_step(model, x, y, optimizer, use_amp, scaler):
    optimizer.zero_grad()

    if use_amp:
        return train_step_amp(model, x, y, optimizer, scaler)
    return train_step_normal(model, x, y, optimizer)


def train_step_amp(model, x, y, optimizer, scaler):
    with torch.cuda.amp.autocast():
        _, loss = model(x, y)
    scaler.scale(loss).backward()

    if config["Training"]["grad_clip"] is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["Training"]["grad_clip"])

    scaler.step(optimizer)
    scaler.update()
    return loss


def train_step_normal(model, x, y, optimizer):
    _, loss = model(x, y)
    loss.backward()

    if config["Training"]["grad_clip"] is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["Training"]["grad_clip"])

    optimizer.step()
    return loss


def main():
    torch.cuda.empty_cache()

    full_dataset, train_dataset, val_dataset = setup_dataset()
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)
    device, model, optimizer = setup_training(full_dataset)
    lr_scheduler = setup_lr_scheduler(optimizer, train_loader)

    print_training_info(device, model, full_dataset,
                        train_dataset, val_dataset)
    load_best_model(model, device)

    if not os.path.exists(config["Training"]["save_dir"]):
        os.makedirs(config["Training"]["save_dir"])

    best_validation_loss = float('inf')
    no_improvement_count = 0
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(config["Training"]["epochs"]):
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, lr_scheduler,
            device, use_amp, scaler, epoch
        )

        if (epoch + 1) % config["Training"]["eval_every"] == 0:
            val_loss = evaluate(model, val_loader, device, use_amp)

            print("-" * 30)
            print(f"Epoch {epoch+1}/{config['Training']['epochs']}")
            print(f"Average Train Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 30)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                no_improvement_count = 0
                torch.save(model.state_dict(),
                           f"{config['Training']['save_dir']}/best_model.pth")
                print(
                    f"New best model saved! Loss: {best_validation_loss:.4f}")
            else:
                no_improvement_count += 1

            if no_improvement_count >= config["Training"]["patience"]:
                print(
                    f"No improvement in {config['Training']['patience']} evaluations. Early stopping.")
                break

    print("Training completed!")


if __name__ == "__main__":
    main()
