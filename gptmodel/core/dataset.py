from typing import List
import torch
import json
from typing import List, Dict
from transformers import GPT2Tokenizer
from datasets import load_dataset
import tiktoken
from torch.utils.data import Dataset
from typing import Tuple


class TinyShakespeare:
    def __init__(self, file_path, block_size):
        with open(file_path, 'r') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.stoi)
        self.block_size = block_size
        self.data = self.encode(self.text)

    def encode(self, text):
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        return x, y

    def __len__(self):
        return len(self.data)

    def decode(self, x):
        return ''.join([self.itos[i] for i in x])


class AlpacaDataset(Dataset):
    def __init__(self, file_path: str, block_size: int):
        super().__init__()
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-2")

        # Ensure block_size doesn't exceed model's maximum position embeddings
        self.block_size = min(block_size, 1024)

        # Load and process the Alpaca dataset
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        # Process conversations
        self.tokenized_data = []
        for item in self.raw_data:
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item['output']

            # Format the conversation more concisely in a full string format
            if input_text:
                conversation = f"Question: {instruction} {input_text}\nAnswer: {output}\n\n"
            else:
                conversation = f"Question: {instruction}\nAnswer: {output}\n\n"

            # Tokenize without truncation parameter
            tokens = self.tokenizer.encode(conversation)

            # Split into chunks of block_size with overlap
            for i in range(0, len(tokens), self.block_size // 2):
                chunk = tokens[i:i + self.block_size]

                # Only add if chunk is long enough to be meaningful
                if len(chunk) > self.block_size // 4:
                    # Pad if needed
                    if len(chunk) < self.block_size:
                        chunk = chunk + \
                            [0] * \
                            (self.block_size - len(chunk))
                    # Truncate if needed
                    chunk = chunk[:self.block_size]
                    self.tokenized_data.append(
                        torch.tensor(chunk, dtype=torch.long))

        self.vocab_size = self.tokenizer.n_vocab

        print(f"Loaded {len(self.tokenized_data)} blocks")
        print(f"Maximum sequence length: {self.block_size}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example (input, target) pair
        """
        tokens = self.tokenized_data[idx]
        x = tokens[:-1]  # all but last token
        y = tokens[1:]   # all but first token
        return x, y

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text with truncation to block_size"""
        return torch.tensor(self.tokenizer.encode(text))

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text, skipping special tokens"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
