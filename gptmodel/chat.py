import torch
from colorama import Fore, Style
from core.model import GPTlite
from core.dataset import AlpacaDataset
from core.utils import load_model
from config import config


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def initialize_model(model_path, dataset_path):
    device = get_device()
    dataset = AlpacaDataset(dataset_path, config["model"]["block_size"])

    model_config = {
        'context_size': config["model"]["block_size"],
        'vocab_size': dataset.vocab_size,
        'embedding_dim': config["model"]["n_embd"],
        'num_heads': config["model"]["n_head"],
        'num_layers': config["model"]["n_layer"],
        'dropout': config["model"]["dropout"]
    }

    model = GPTlite(model_config).to(device)
    model.eval()

    return model, dataset, device


def generate_response(model, dataset, prompt, device, max_new_tokens=70):
    input_tensor = dataset.encode(prompt).unsqueeze(0).to(device)

    with torch.no_grad():
        generated_indices = model.generate(input_tensor, max_new_tokens)
        generated_text = dataset.decode(generated_indices[0].tolist())

    return generated_text[len(prompt):]


def format_prompt(user_input):
    return f"Question: {user_input}\nAnswer:"


def chat_loop(model, dataset, device):
    print("Model initialized! Start chatting (type 'quit' to exit)")
    print("-----")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        prompt = format_prompt(user_input)
        print("Assistant: ", end='')
        response = generate_response(model, dataset, prompt, device)
        print(response)
        print("-----")


def main():
    print("Initializing RadioGPT Lite...")
    model_path = f"{config['Training']['save_dir']}/best_model.pth"
    dataset_path = config["dataset"]["file_path"]

    model, dataset, device = initialize_model(model_path, dataset_path)
    chat_loop(model, dataset, device)


if __name__ == "__main__":
    main()
