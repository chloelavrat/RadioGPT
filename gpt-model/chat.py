import torch
from core.model import GPTlite
from core.dataset import AlpacaDataset
from colorama import Fore, Style
from core.utils import load_model


def initialize_model(model_path, dataset_path):
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )

    n_embd = 512
    n_head = 6
    n_layer = 10
    dropout = 0.2
    context_size = 64

    # model = load_model(model_path, device)
    dataset = AlpacaDataset(dataset_path, context_size)

    model = GPTlite({
        'context_size': context_size,
        'vocab_size': dataset.vocab_size,
        'embedding_dim': n_embd,
        'num_heads': n_head,
        'num_layers': n_layer,
        'dropout': dropout
    }).to(device)

    model.eval()

    return model, dataset, device


def generate_response(model, dataset, prompt, device, max_new_tokens=70):
    # Encode the prompt
    input_tensor = dataset.encode(prompt).unsqueeze(0).to(device)

    # Generate text using the model's generate method
    with torch.no_grad():
        generated_indices = model.generate(input_tensor, max_new_tokens)
        generated_text = dataset.decode(generated_indices[0].tolist())

    # Return only the newly generated part (after the prompt)
    return generated_text[len(prompt):]


def chat():
    # Initialize the model
    print(Fore.GREEN + "Initializing ChatGPT Lite..." + Style.RESET_ALL)
    model, dataset, device = initialize_model(
        "gpt-model/models/best_model.pth",
        "gpt-model/dataset/alpaca_data_en.json"
    )
    print(Fore.GREEN + "Model initialized! Start chatting (type 'quit' to exit)" + Style.RESET_ALL)
    print("-----")

    while True:
        user_input = input(Fore.BLUE + "You: " + Style.RESET_ALL)
        if user_input.lower() == 'quit':
            break

        # Add a simple prompt template
        prompt = f"Question: {user_input}\nAnswer:"

        print(Fore.GREEN + "Assistant: " + Style.RESET_ALL, end='')
        response = generate_response(model, dataset, prompt, device)
        print(response)
        print("-----")


if __name__ == "__main__":
    chat()
