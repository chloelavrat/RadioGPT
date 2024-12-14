<div align="center">
  <img src="https://openfileserver.chloelavrat.com/workshops/RadioGPT/assets/radiogpt-banner.png" alt="Banner" style="border-radius: 17px; width: 100%; max-width: 800px; height: auto;">
</div>

<h3 align="center">
  <b><a href="https://torch-crepe-demo.chloelavrat.com">Interactive Demo</a></b>
  •
  <b><a href="https://www.youtube.com">Video</a></b>
  •
  <b><a href="">Python API</a></b>
</h3>

<div align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/chloelavrat/RadioGPT/actions/workflows/CI.yml">
    <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python Versions">
  </a>
</div>

<p align="center">The <b>RadioGPT</b> project is an educational project that demonstrates how to build and train language models from scratch. Through a series of progressive notebooks, you'll learn how to create increasingly sophisticated language models, from a basic character-level model to a more advanced chat-capable LLM.</p>


## Project Structure
The project consists of three main notebooks, each building upon the previous one:

1. **RadioGPT_1_Generateur_de_Moliere.ipynb**
   - Introduction to basic language model concepts
   - Character-level tokenization
   - Training a simple model on Molière's works
   - Understanding fundamental LLM components
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloelavrat/RadioGPT/blob/main/RadioGPT_1_Generateur_de_Moliere.ipynb)

2. **RadioGPT_2_Larger_LLM_Chat.ipynb**
   - Playing with a larger model (83.1M parameters)
   - Working with more sophisticated architectures
   - Improved text generation capabilities
   - Chat-oriented model training
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloelavrat/RadioGPT/blob/main/RadioGPT_2_Larger_LLM_Chat.ipynb)
3. **RadioGPT_3_Finally_RadioGPT.ipynb**
   - Fine-tuning on radio station data
   - Advanced model architecture
   - Real-world application with radio content
   - Multiple radio station dataset options
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chloelavrat/RadioGPT/blob/main/RadioGPT_3_Finally_RadioGPT.ipynb)
## Core Components

### Model Architecture (`gptmodel/core/model.py`)
- `GPTlite`: A lightweight GPT-style transformer
- Modular attention mechanism
- Scalable architecture with configurable parameters
- Support for both training and generation

### Dataset Handling (`gptmodel/core/dataset.py`)
- Multiple dataset classes:
  - `TinyShakespeare`: Character-level dataset
  - `AlpacaDataset`: Chat-oriented dataset

### Dataset links
- `TinyShakespeare` style:
  - [petit molière](https://openfileserver.chloelavrat.com/workshops/RadioGPT/dataset/petitmoliere.txt)
- `ALpaca` style:
  - [Acquiesce_data_110k_instructions](https://openfileserver.chloelavrat.com/workshops/RadioGPT/dataset/Acquiesce_data_110k_instructions.json)

### Utilities (`gptmodel/core/utils.py`)
- Training and evaluation functions
- Model saving and loading
- Text generation utilities
- Performance monitoring

## Requirements
```
pip install torch datasets tqdm transformers
```
- GPU recommended on Google Colab (T4 or better)

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RadioGPT.git
```

2. Install dependencies:
```bash
pip install torch datasets tqdm transformers
```

3. Open the notebooks in order:
   - Start with `RadioGPT_1_Generateur_de_Moliere.ipynb`
   - Progress to `RadioGPT_2_Larger_LLM_Chat.ipynb`
   - Finally, explore `RadioGPT_3_Finally_RadioGPT.ipynb`

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes. This has been developed in less than 6 days, so there is a lot of room for improvement. 😉

## License
- Developped by **Chloé Lavrat** for **Radio France**
- **All rights reserved by Radio France and Chloé Lavrat**

## Acknowledgments
- Thanks to **Marc Yefimchuk** and **Jade Moillic** for the help in processing the data and improving the notebooks
- Thanks to the "attention is all you need" authors and the GPT-2 authors for the inspiration 🥰
- Thanks to the **PyTorch team** for the excellent deep learning framework
- Special thanks to contributors and the community
