import torch
import os
import subprocess


def download_dataset(url, destination):
    print("Downloading dataset...")
    os.makedirs("dataset", exist_ok=True)
    subprocess.run(["wget", url, "-O", destination])
    print("Dataset downloaded!")


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


def load_model(model_path, device, config):
    # Load the model checkpoint
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False)

    model = GPTlite(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
