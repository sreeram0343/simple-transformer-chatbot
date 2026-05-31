# 🔮 Simple Transformer Chatbot (Neural Orchestrator)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A scratch-built, decoder-only GPT-style Transformer chatbot implemented in PyTorch. This project features a sophisticated **Neural Cyber-Noir** dashboard for training, real-time metrics, and interactive chat.

---

## 🌟 Project Philosophy

This repository is designed as a deep-dive learning resource. We implement every mathematical component of the Transformer architecture manually using **Python** and **PyTorch**, bridging the gap between theoretical papers and production-grade code.

### What makes this project unique?
- **Zero-Abstraction**: No `transformers` library. Just raw tensors and logic.
- **Neural Cyber-Noir UI**: A high-tech, immersive dashboard built with CustomTkinter.
- **Advanced Generation**: Supports **Top-k sampling** and temperature control.
- **End-to-End**: Covers everything from raw text tokenization to autoregressive inference.

## 🚀 Core Features

- **Custom Tokenizer**: Word-level encoding with special token handling (`<SEP>`, `<EOS>`, `<PAD>`).
- **Transformer Architecture**: Implementation of Multi-Head Attention, Feed-Forward Networks, and Layer Normalization.
- **Neural Cyber-Noir Dashboard**: A full-featured desktop interface for training and chatting.
- **Real-time Analytics**: Visualization of loss gradients and attention focus heatmaps.
- **Unified Logging**: Professional logging system for robust debugging.
- **Comprehensive Testing**: Automated unit tests for core NLP components.

## 🏗️ Architectural Highlights

The model follows the modern "Pre-LN" Transformer variant, ensuring stable training and better gradient flow.

| Component | Description |
| :--- | :--- |
| **Causal Masking** | Prevents the model from "cheating" by looking at future tokens during training. |
| **Weight Tying** | Shares parameters between the embedding layer and the output projection for efficiency. |
| **Positional Encoding** | Uses sinusoidal functions to provide the model with a sense of token order. |
| **Top-k Sampling** | Introduces controlled randomness for more natural responses. |

---

## 📂 Project Structure

```bash
.
├── data/               # Training data (conversations.txt)
├── src/                # Core Transformer implementation
│   ├── model.py        # Main Transformer class
│   ├── tokenizer.py    # Text processing logic
│   └── chat.py         # Inference pipeline & CLI
├── tests/              # Automated unit tests
├── ui/                 # Neural Cyber-Noir Dashboard
│   ├── app.py          # Main application entry
│   └── ...             # UI panels and styles
├── train.py            # CLI Training script
└── README.md           # Documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+**
- **pip** (Python package manager)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/sreeram0343/simple-transformer-chatbot.git
cd simple-transformer-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### Option A: The Neural Orchestrator Dashboard (GUI)
Experience the full power of the project with a modern graphical interface.
```bash
python ui/app.py
```

### Option B: Command Line Interface (CLI)
For those who prefer the terminal.
```bash
# Train the model first
python train.py

# Start chatting
python src/chat.py
```

---

## 🧪 Testing
Run the automated test suite to verify the core components:
```bash
python -m unittest discover tests
```

---

## 🤝 Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

Happy Learning! You've built a GPT-like generative bot entirely from scratch!
