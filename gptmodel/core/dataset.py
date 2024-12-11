from typing import List
import torch
import json
from typing import List, Dict
from transformers import GPT2Tokenizer
from datasets import load_dataset


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


class AlpacaDataset:
    def __init__(self, file_path: str, block_size: int):
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure block_size doesn't exceed model's maximum position embeddings
        self.block_size = min(block_size, 1024)

        # Load and process the Alpaca dataset
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        # Process conversations
        self.conversations = []
        self.tokenized_data = []
        for item in self.raw_data:
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item['output']

            # Format the conversation more concisely
            if input_text:
                conversation = f"Question: {instruction} {input_text}\nAnswer: {output}\n\n"
            else:
                conversation = f"Question: {instruction}\nAnswer: {output}\n\n"

            # Truncate long conversations at tokenization time
            tokens = self.tokenizer.encode(
                conversation, truncation=True, max_length=self.block_size)
            if len(tokens) > 0:  # Only add if we got tokens
                self.tokenized_data.append(
                    torch.tensor(tokens, dtype=torch.long))

        self.vocab_size = self.tokenizer.vocab_size
        print(f"Loaded {len(self.tokenized_data)} conversations")
        print(f"Maximum sequence length: {self.block_size}")

        # Check for invalid entries in the raw data
        for item in self.raw_data:
            if 'instruction' not in item or 'output' not in item:
                print("Invalid entry found in dataset:", item)

    def get_batch(self, batch_size: int):
        # Randomly select conversations
        conv_indices = torch.randint(
            0, len(self.tokenized_data), (batch_size,))
        x_list = []
        y_list = []

        for idx in conv_indices:
            tokens = self.tokenized_data[idx]

            # If conversation is longer than block_size, randomly select a starting point
            if len(tokens) >= self.block_size:
                start_idx = torch.randint(
                    0, len(tokens) - self.block_size + 1, (1,))
                chunk = tokens[start_idx:start_idx + self.block_size]
            else:
                # If conversation is shorter, pad it
                chunk = torch.cat([
                    tokens,
                    torch.full((self.block_size - len(tokens),),
                               self.tokenizer.pad_token_id)
                ])

            # Ensure chunk is exactly block_size
            chunk = chunk[:self.block_size]

            # Input is all tokens except last, target is all tokens except first
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        return x, y

    def encode(self, text: str) -> torch.Tensor:
        """Encode text with truncation to block_size"""
        return torch.tensor(
            self.tokenizer.encode(text, truncation=True,
                                  max_length=self.block_size),
            dtype=torch.long
        )

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text, skipping special tokens"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def __len__(self) -> int:
        return len(self.tokenized_data)


class SQuADDataset:
    def __init__(self, file_path: str, block_size: int):
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure block_size doesn't exceed model's maximum position embeddings
        self.block_size = min(block_size, 1024)

        # Load and process the SQuAD dataset
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        # Process questions and context
        self.tokenized_data = []
        for article in self.raw_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = qa.get('answers', [])
                    # Check if answers list is not empty
                    if answers:
                        answer = answers[0].get('text', '')
                        # Create the conversation-like structure
                        conversation = f"Question: {context} {question}\nAnswer: {answer}\n\n"
                        # Truncate long conversations at tokenization time
                        tokens = self.tokenizer.encode(
                            conversation, truncation=True, max_length=self.block_size)
                        if len(tokens) > 0:  # Only add if we got tokens
                            self.tokenized_data.append(
                                torch.tensor(tokens, dtype=torch.long))

        self.vocab_size = self.tokenizer.vocab_size
        print(f"Loaded {len(self.tokenized_data)} question-answer pairs")
        print(f"Maximum sequence length: {self.block_size}")

    def get_batch(self, batch_size: int):
        # Randomly select conversations
        conv_indices = torch.randint(
            0, len(self.tokenized_data), (batch_size,))
        x_list = []
        y_list = []

        for idx in conv_indices:
            tokens = self.tokenized_data[idx]

            # If conversation is longer than block_size, randomly select a starting point
            if len(tokens) >= self.block_size:
                start_idx = torch.randint(
                    0, len(tokens) - self.block_size + 1, (1,))
                chunk = tokens[start_idx:start_idx + self.block_size]
            else:
                # If conversation is shorter, pad it
                chunk = torch.cat([
                    tokens,
                    torch.full((self.block_size - len(tokens),),
                               self.tokenizer.pad_token_id)
                ])

            # Ensure chunk is exactly block_size
            chunk = chunk[:self.block_size]

            # Input is all tokens except last, target is all tokens except first
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        return x, y

    def encode(self, text: str) -> torch.Tensor:
        """Encode text with truncation to block_size"""
        return torch.tensor(
            self.tokenizer.encode(text, truncation=True,
                                  max_length=self.block_size),
            dtype=torch.long
        )

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text, skipping special tokens"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def __len__(self) -> int:
        return len(self.tokenized_data)


class OpenWebTextDataset:
    def __init__(self, block_size: int):
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure block_size doesn't exceed model's maximum position embeddings
        self.block_size = min(block_size, 1024)

        # Load OpenWebText dataset
        self.dataset = load_dataset("openwebtext", num_proc=8)
        self.tokenized_data = []

        # Process and tokenize the texts
        print("Tokenizing OpenWebText dataset...")
        for item in self.dataset['train']:
            tokens = self.tokenizer.encode(
                item['text'], truncation=True, max_length=self.block_size)
            if len(tokens) > 0:
                self.tokenized_data.append(torch.tensor(tokens, dtype=torch.long))

        self.vocab_size = self.tokenizer.vocab_size
        print(f"Loaded {len(self.tokenized_data)} text samples")
        print(f"Maximum sequence length: {self.block_size}")

    def get_batch(self, batch_size: int):
        # Randomly select texts
        indices = torch.randint(0, len(self.tokenized_data), (batch_size,))
        x_list = []
        y_list = []

        for idx in indices:
            tokens = self.tokenized_data[idx]

            # If text is longer than block_size, randomly select a starting point
            if len(tokens) >= self.block_size:
                start_idx = torch.randint(0, len(tokens) - self.block_size + 1, (1,))
                chunk = tokens[start_idx:start_idx + self.block_size]
            else:
                # If text is shorter, pad it
                chunk = torch.cat([
                    tokens,
                    torch.full((self.block_size - len(tokens),),
                              self.tokenizer.pad_token_id)
                ])

            # Ensure chunk is exactly block_size
            chunk = chunk[:self.block_size]

            # Input is all tokens except last, target is all tokens except first
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        return x, y

    def encode(self, text: str) -> torch.Tensor:
        """Encode text with truncation to block_size"""
        return torch.tensor(
            self.tokenizer.encode(text, truncation=True,
                                max_length=self.block_size),
            dtype=torch.long
        )

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text, skipping special tokens"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def __len__(self) -> int:
        return len(self.tokenized_data)
