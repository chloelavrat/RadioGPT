import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from RadioGPT.gptmodel.core.model import GPTlite


def train(model, dataset, optimizer, epochs, batch_size, device,
          patience=500, evaluation_epoch=200, save_dir="models",
          grad_clip=1.0, warmup_iters=0, lr_scheduler=None,
          learning_rate=6e-4):
    import os
    model.train()
    best_validation_loss = float('inf')
    no_improvement_count = 0
    total_training_loss = 0.0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    progress_bar = tqdm(range(epochs), desc="Training", unit="step")

    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for step in progress_bar:
        # Get batch
        indices, targets = dataset.get_batch(batch_size)
        indices, targets = indices.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            logits, loss = model(indices, targets)

        # Scale loss and backprop
        scaler.scale(loss).backward()

        # Gradient clipping
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Learning rate scheduling
        if lr_scheduler is not None:
            if warmup_iters > 0 and step < warmup_iters:
                # Linear warmup
                lr = learning_rate * (step + 1) / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif step < epochs:  # Ensure we only step if within the limit
                lr_scheduler.step()

        total_training_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

        # Add loss monitoring and early stopping
        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"NaN or Inf loss detected at step {step}. Stopping training.")
            break

        # Monitor gradient norms
        if step % evaluation_epoch == 0:
            avg_loss = total_training_loss / \
                (evaluation_epoch if step > 0 else 1)
            print("-" * 30)
            print(f"Step {step}/{epochs}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

            # Evaluate
            eval_loss = evaluate(model, dataset, device)
            print(f"Validation Loss: {eval_loss:.4f}")
            print("-" * 30)

            if eval_loss < best_validation_loss:
                best_validation_loss = eval_loss
                no_improvement_count = 0
                torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
                print(
                    f"New best model saved! Loss: {best_validation_loss:.4f}")
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                print(
                    f"No improvement in {patience} evaluations. Early stopping.")
                break

            total_training_loss = 0.0

    torch.save(model.state_dict(), f"{save_dir}/last_model.pth")


def generate_response(model, dataset, prompt, device, max_new_tokens=70):
    # Encode the prompt
    input_tensor = dataset.encode(prompt).unsqueeze(0).to(device)

    # Generate text using the model's generate method
    with torch.no_grad():
        generated_indices = model.generate(input_tensor, max_new_tokens)
        generated_text = dataset.decode(generated_indices[0].tolist())

    # Return only the newly generated part (after the prompt)
    return generated_text[len(prompt):]


def evaluate(model, dataset, device, num_batches=10, batch_size=32):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(num_batches):
            idx, targets = dataset.get_batch(batch_size)
            idx, targets = idx.to(device), targets.to(device)
            logits, loss = model(idx, targets)
            total_loss += loss.item()

    return total_loss / num_batches


def save_model(model, save_dir="models"):
    config = {
        'vocab_size': model.token_embeddings.num_embeddings,
        'embedding_dim': model.token_embeddings.embedding_dim,
        'num_heads': len(model.blocks[0].attention.attention_heads),
        'num_layers': len(model.blocks),
        'context_size': model.context_size,
        'dropout': model.blocks[0].attention.dropout.p
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, f"{save_dir}/model.pth")


def load_model(model_path, device):
    # Load the model checkpoint
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)

    # Check available keys in the checkpoint
    for key in checkpoint:
        if key == 'config':
            config = checkpoint[key]
            break

    if config is None:
        raise KeyError("'config' key not found in the checkpoint. Available keys: {}".format(
            checkpoint.keys()))

    # Ensure 'config' key exists
    if 'config' not in checkpoint:
        raise KeyError("'config' key not found in the checkpoint. Available keys: {}".format(
            checkpoint.keys()))

    config = checkpoint['config']
    model = GPTlite(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
