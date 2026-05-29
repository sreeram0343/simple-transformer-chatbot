# 🔮 Simple Transformer Chatbot From Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)](https://github.com/sreeram0343/simple-transformer-chatbot)

A professional, educational implementation of a **Generative Decoder-Only Transformer** (GPT-style) built entirely from the ground up. This project avoids high-level abstractions like Hugging Face to provide a transparent look into the mechanics of modern NLP.

---

## 🌟 Project Philosophy

This repository is designed as a deep-dive learning resource. We implement every mathematical component of the Transformer architecture manually using **Python** and **PyTorch**, bridging the gap between theoretical papers and production-grade code.

### What makes this project unique?
- **Zero-Abstraction**: No `transformers` library. Just raw tensors and logic.
- **Interactive Visuals**: Real-time attention heatmaps and training monitoring.
- **End-to-End**: Covers everything from raw text tokenization to autoregressive inference.

## 🚀 Core Features

- **Custom Tokenizer**: Word-level encoding with special token handling (`<SEP>`, `<EOS>`, `<PAD>`).
- **Transformer Architecture**: implementation of Multi-Head Attention, Feed-Forward Networks, and Layer Normalization.
- **GUI Dashboard**: A full-featured desktop interface for training and chatting.
- **Real-time Analytics**: Visualization of loss curves and attention weight heatmaps.
- **Inference Pipeline**: Autoregressive sequence generation with greedy search.

## 🏗️ Architectural Highlights

The model follows the modern "Pre-LN" Transformer variant, ensuring stable training and better gradient flow.

| Component | Description |
| :--- | :--- |
| **Causal Masking** | Prevents the model from "cheating" by looking at future tokens during training. |
| **Weight Tying** | Shares parameters between the embedding layer and the output projection for efficiency. |
| **Positional Encoding** | Uses sinusoidal functions to provide the model with a sense of token order. |
| **Pre-LayerNorm** | Applies normalization before the attention and FFN blocks for improved stability. |

---

## 📂 Project Structure

```bash
.
├── data/               # Training data (conversations.txt)
├── src/                # Core Transformer implementation
│   ├── model.py        # Main Transformer class
│   ├── tokenizer.py    # Text processing logic
│   └── ...             # Modular components (Attention, PE, etc.)
├── ui/                 # GUI Dashboard (CustomTkinter)
│   ├── app.py          # Main application entry
│   └── ...             # UI panels and styles
├── train.py            # CLI Training script
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install PyTorch:
```bash
pip install -r requirements.txt
```

### 2. Train the Chatbot (Stages 6 - 9)
Train the custom Transformer on our conversations dataset by running the master training loop. It will train for 40 epochs, display loss progression, and serialize weights to `chatbot_model.pt` and vocabulary keys to `vocab.json`:
```bash
python train.py
```

### 3. Evaluate the Model (Stage 13)
Quantify model performance and check if the training succeeded by running the evaluation script:
```bash
python src/evaluate.py
```
This will report:
* **Average Loss**
* **Perplexity (PPL)**
* **Next-Token Prediction Accuracy**

### 4. Chat with the Chatbot (Stages 10 - 12)
Launch the CLI interface to talk with your trained model. The chatbot uses **greedy autoregressive decoding** and possesses a **conversation memory buffer**:
```bash
python src/chat.py
```
* Type `clear` to wipe conversation memory.
* Type `exit` to quit.

---

## 💡 Key Architectural Details to Remember

* **Trigonometric PE**: Standard sinusoidal curves are deterministic and allow training sequences to scale larger.
* **Causal Masking**: Upper-triangular elements are filled with $-\infty$ so the softmax ignores future look-ahead tokens, enforcing causal unidirectional generation.
* **Pre-Layer Normalization (Pre-LN)**: Norm is applied before attention and FFN to ensure smooth gradient flow through deep networks.
* **Weight Tying**: We tied `token_embeddings.weight` directly to `lm_head.weight`. This binds input mapping and output projection parameters, cutting model size down and preventing overfitting on small vocabularies.
* **Instruction-Tuning Target Masking**: Setting user tokens to `-100` targets allows PyTorch's `CrossEntropyLoss` to bypass calculating gradients for predicting user inputs, training the model solely to formulate bot answers.

Happy Learning! You've built a GPT-like generative bot entirely from scratch!
