# Project Architecture

This document provides a detailed overview of the technical design and data flow of the Simple Transformer Chatbot.

## System Overview

The project is divided into three main layers:
1. **Core Layer (`src/`)**: Contains the manual implementation of the Transformer architecture.
2. **Application Layer (`train.py`, `src/chat.py`, `src/evaluate.py`)**: Handles model training, interactive inference, and performance metrics.
3. **UI Layer (`ui/`)**: A graphical dashboard built with CustomTkinter for a user-friendly experience.

## Data Flow

### 1. Training Phase
1. **Raw Text**: `data/conversations.txt` stores user-bot pairs.
2. **Tokenization**: `Tokenizer` builds a vocabulary and converts text to integer IDs.
3. **Dataset**: `ConversationDataset` prepares batches, concatenates sequences, and masks user input.
4. **Model**: `TransformerChatbot` processes inputs through stacked blocks.
5. **Optimization**: `AdamW` optimizer updates weights based on `CrossEntropyLoss`.

### 2. Inference Phase (Chat)
1. **User Input**: Received via CLI or GUI.
2. **Encoding**: Text is converted to IDs and appended to conversation history.
3. **Autoregression**: The model predicts the next token one-by-one until `<EOS>` is reached.
4. **Decoding**: Token IDs are converted back to readable text.

## Core Components

- **Multi-Head Attention**: Allows the model to attend to different parts of the sequence simultaneously.
- **Positional Encoding**: Uses fixed sinusoidal functions to provide token position information.
- **Weight Tying**: Reduces parameter count and improves generalization by sharing weights between input and output layers.

## Technology Stack

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch 2.0+
- **GUI**: CustomTkinter
- **Mathematics**: NumPy
- **Visualization**: Matplotlib
