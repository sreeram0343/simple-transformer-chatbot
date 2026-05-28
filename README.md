# 🔮 Custom Transformer Chatbot From Scratch

Welcome to the **Transformer Chatbot From Scratch**! This is a complete learning project designed to teach the inner workings of generative decoder-only Transformer models (like the GPT series) by implementing every component manually using **Python** and **PyTorch**—without using any ready-made libraries like Hugging Face.

---

## 🗺️ The 14-Stage Learning Journey

This project is built step-by-step across 14 developmental stages:

1. **Positional Encoding** 📍: Sine & cosine waves to add word order signals into sequential embeddings.
2. **Self-Attention** 🔍: The core mechanism allowing words to contextualize themselves by attending to other words.
3. **Multi-Head Attention** 🧠: Running multiple self-attention heads in parallel to focus on different aspects of the text.
4. **Transformer Block** 🧱: Combining Multi-Head Attention, Pre-Layer Normalization, Residual Connections, and a Position-wise Feed-Forward Network.
5. **Dataset Preparation** 📊: Structuring user-bot sentences with special `<SEP>` and `<EOS>` tokens and masking user prompts with `-100` targets.
6. **Training Loop** 🔄: Ship parameters to GPU/CPU, compile batches, and iterate.
7. **Loss Function** 📉: Aligning shifted next-token logits with Cross-Entropy Loss ignoring `-100` targets.
8. **Backpropagation** ⚡: Zeroing gradients, calculating weight derivatives (`loss.backward()`), and running step optimizations.
9. **Saving and Loading** 💾: Storing model weights as `.pt` and vocabulary dicts as `.json`.
10. **Inference Pipeline** 🔮: Autoregressive token-by-token sequence generation using Greedy Search.
11. **Chat Interface** 💬: Interactive command-line loop.
12. **Conversation Memory** 💭: Context retention by maintaining history logs formatted as `User <SEP> Bot <EOS> ...`.
13. **Evaluation** 📈: Quantifying performance using Average Cross-Entropy, Perplexity ($e^{\text{loss}}$), and Next-Token Accuracy.
14. **Project Refactoring** 🏗️: Standardizing requirements, directory separation, and detailed documentation.

---

## 📂 Project Structure

```
transformer-chatbot/
│
├── data/
│   └── conversations.txt      # Training conversation pairs (User|Bot)
│
├── src/
│   ├── tokenizer.py           # Word-level encoder/decoder & file loading
│   ├── positional_encoding.py  # Trigonometric PositionalEncoding module
│   ├── self_attention.py       # Single-head scaled dot-product attention
│   ├── multi_head_attention.py # Parallel attention heads
│   ├── transformer_block.py   # GPT-style Decoder block with residual connections
│   ├── dataset.py             # PyTorch dataset mapping and label masking (-100)
│   ├── model.py               # Combined Transformer Chatbot & Weight Tying
│   ├── chat.py                # Inference pipeline, CLI, and conversation memory
│   └── evaluate.py            # Computes loss, perplexity, and token accuracy
│
├── train.py                   # Master training loop, AdamW optimizer, saving weights
├── requirements.txt           # Core Python dependencies
└── README.md                  # Master documentation
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
