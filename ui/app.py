import sys
import os
import threading
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import customtkinter as ctk

# Ensure correct importing from both root, src, and local ui/ directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import styles as st
from sidebar import SidebarPanel
from chat_panel import ChatPanel
from training_panel import TrainingPanel
from visualization_panel import VisualizationPanel

# Import backend classes from parent directory
from src.tokenizer import Tokenizer, load_conversations
from src.dataset import ConversationDataset
from src.model import TransformerChatbot
from src.chat import ChatbotPipeline


class TrainingWorker(threading.Thread):
    """
    Background Thread Worker executing the Transformer training loop (Stage 6).
    
    Why it exists:
    Training a deep learning model is highly resource-intensive. If we run it in the main
    thread, the desktop GUI will instantly freeze (not responding). Running it in a
    separate thread keeps the GUI completely smooth, allowing real-time loss updates.
    """
    def __init__(self, app, epochs=40, lr=0.001, d_model=32, num_heads=4, d_ff=128, num_layers=2, max_len=32):
        super().__init__()
        self.app = app
        self.epochs = epochs
        self.lr = lr
        
        # Architecture parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        
        self.is_active = True  # Flag to support early stopping

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.app.safe_update_status("training")
            
            # 1. Dataset Loading
            data_path = "data/conversations.txt"
            pairs = load_conversations(data_path)
            
            flat_texts = [text for pair in pairs for text in pair]
            tokenizer = Tokenizer()
            tokenizer.train(flat_texts)
            vocab_size = len(tokenizer.vocab)
            
            # Save vocabulary configuration immediately for inference
            with open("vocab.json", "w", encoding="utf-8") as f:
                json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=4)
                
            dataset = ConversationDataset(pairs=pairs, tokenizer=tokenizer, max_len=self.max_len)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            # 2. Model compilation
            model = TransformerChatbot(
                vocab_size=vocab_size,
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                num_layers=self.num_layers,
                max_len=self.max_len
            )
            model.to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            
            model.train()
            
            # 3. Step Epoch Loop
            for epoch in range(1, self.epochs + 1):
                # Check for stop trigger clicked by user
                if not self.is_active:
                    print("[*] Background Worker: Stop request received. Terminating training.")
                    break
                    
                total_loss = 0.0
                
                for batch in dataloader:
                    if not self.is_active:
                        break
                        
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    seq_len = input_ids.size(1)
                    mask_template = torch.tril(torch.ones(seq_len, seq_len, device=device))
                    causal_mask = torch.zeros(seq_len, seq_len, device=device).masked_fill(mask_template == 0, float('-inf'))
                    
                    logits = model(input_ids, mask=causal_mask)
                    
                    shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                    shift_labels = labels[:, 1:].contiguous().view(-1)
                    
                    loss = criterion(shift_logits, shift_labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                if not self.is_active:
                    break
                    
                avg_loss = total_loss / len(dataloader)
                
                # Safely update GUI thread (epoch completed)
                # .after(0, ...) pushes the callback into Tkinter's main event queue
                self.app.after(0, self.app.on_epoch_complete, epoch, self.epochs, avg_loss)
                time.sleep(0.02)  # Muted sleep to prevent visual clipping
                
            if self.is_active:
                # Save trained parameters (Stage 9)
                torch.save(model.state_dict(), "chatbot_model.pt")
                self.app.after(0, self.app.on_training_finished, True)
            else:
                self.app.after(0, self.app.on_training_finished, False)
                
        except Exception as e:
            print(f"[ERROR] Training thread failure: {e}")
            self.app.after(0, self.app.on_training_failed, str(e))

    def stop(self):
        """Triggers early cancellation of training loop."""
        self.is_active = False


class TransformerChatbotApp(ctk.CTk):
    """
    Main Tkinter GUI Window Application connecting frontend views with backend threads.
    """
    def __init__(self):
        super().__init__()
        
        # 1. GENERAL CUSTOMTKINTER INITIALIZATION
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.title("🔮 NEURAL ORCHESTRATOR :: CORE INTERFACE")
        self.geometry("1100x720")
        self.minsize(1000, 640)
        self.configure(fg_color=st.BG_MAIN)
        
        # 2. CONFIGURE RESPONSIVE GRID LAYOUT
        # Left column (sidebar) is fixed, right column (tabs) scales fully
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Active workers and pipeline state
        self.training_thread = None
        self.pipeline = None
        self.current_status = "offline"
        
        # 3. INITIALIZE SIDEBAR (Left)
        # Binds clear memory and reload action callbacks
        self.sidebar = SidebarPanel(
            self, 
            on_clear_callback=self.action_clear_memory, 
            on_reload_callback=self.action_reload_model
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # 4. INITIALIZE MAIN BODY TABVIEW (Right)
        # Rather than standard page replacements, a central tabbed frame is extremely clean
        self.main_tabview = ctk.CTkTabview(
            self,
            fg_color=st.BG_MAIN,
            segmented_button_fg_color=st.BG_PANEL,
            segmented_button_selected_color=st.ACCENT_PRIMARY_CONTAINER,
            segmented_button_selected_hover_color=st.ACCENT_HOVER,
            segmented_button_unselected_color=st.BG_CARD,
            text_color=st.TEXT_MAIN,
            corner_radius=st.CORNER_RADIUS_LG
        )
        self.main_tabview.grid(row=0, column=1, padx=20, pady=(15, 20), sticky="nsew")
        
        # Add dashboards
        self.tab_chat = self.main_tabview.add("💬 NEURAL CHAT")
        self.tab_train = self.main_tabview.add("🔄 MODEL SYNTHESIS")
        self.tab_visual = self.main_tabview.add("📊 CORE INSIGHTS")
        
        # Configure expansion for tabs
        for tab in [self.tab_chat, self.tab_train, self.tab_visual]:
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            
        # 5. CONSTRUCT COMPONENT PANELS INSIDE TABS
        # Chat Panel
        self.chat_panel = ChatPanel(self.tab_chat, on_send_callback=self.action_send_message)
        self.chat_panel.grid(row=0, column=0, sticky="nsew")
        
        # Training Panel
        self.training_panel = TrainingPanel(
            self.tab_train, 
            on_start_training=self.action_start_training, 
            on_stop_training=self.action_stop_training
        )
        self.training_panel.grid(row=0, column=0, sticky="nsew")
        
        # Visualization Panel
        self.visual_panel = VisualizationPanel(self.tab_visual)
        self.visual_panel.grid(row=0, column=0, sticky="nsew")
        
        # 6. ATTEMPT INITIAL WEIGHT LOAD
        self.action_reload_model()

    # ==========================================
    # CORE INTERFACE CONTROLS & ACTIONS
    # ==========================================
    def action_reload_model(self):
        """Attempts to load trained checkpoint weights and configure pipeline."""
        model_path = "chatbot_model.pt"
        vocab_path = "vocab.json"
        
        if os.path.exists(model_path) and os.path.exists(vocab_path):
            try:
                self.pipeline = ChatbotPipeline(model_path=model_path, vocab_path=vocab_path)
                
                # Extract hyper specs
                vocab_size = len(self.pipeline.tokenizer.vocab)
                device = str(self.pipeline.device)
                
                self.sidebar.update_model_info(
                    device=device,
                    vocab_size=vocab_size,
                    layers=2,
                    embed_dim=32,
                    status="ready"
                )
                self.current_status = "ready"
                print("[+] Interface: Model reload successful.")
            except Exception as e:
                print(f"[!] Interface: Failed to load pipeline files: {e}")
                self.safe_update_status("offline")
        else:
            self.safe_update_status("offline")

    def action_clear_memory(self):
        """Wipes conversation memory log on pipeline and panels."""
        if self.pipeline is not None:
            self.pipeline.clear_memory()
        self.chat_panel.clear_chat()

    # ==========================================
    # INTERACTION GENERATION PIPELINE
    # ==========================================
    def action_send_message(self, text: str):
        """Triggered when send is clicked. Executes inference in background thread."""
        if self.pipeline is None:
            self.chat_panel.add_message("Bot", "⚠️ No model loaded. Please go to the 'Model Training' tab and click 'Start Training' first!")
            return
            
        self.chat_panel.show_typing_indicator()
        
        # Run in background to keep UI fully fluid and support typing visual animations
        threading.Thread(target=self._async_inference, args=(text,), daemon=True).start()

    def _async_inference(self, text: str):
        """Runs autoregressive greedy inference in worker thread."""
        try:
            # Sleep slightly to let the "thinking..." animation display nicely
            time.sleep(0.5)
            
            # Generate token sequence
            response = self.pipeline.generate_response(text)
            
            # Extract internal activation representations for visualizers!
            # 1. Last Attention Matrix: shape (batch, heads, seq, seq)
            attn_weights = self.pipeline.model.blocks[-1].mha.last_attention_weights
            # Head 0, Batch 0: shape (seq_len, seq_len)
            matrix = attn_weights[0, 0, :, :].cpu().numpy()
            
            # 2. Extract parsed string word tokens
            seq_ids = self.pipeline.last_input_ids
            tokens = [self.pipeline.tokenizer.inv_vocab.get(tid, "<UNK>") for tid in seq_ids]
            
            # Update GUI safely in main thread loop
            self.after(0, self._sync_inference_complete, response, matrix, tokens, seq_ids)
            
        except Exception as e:
            print(f"[ERROR] Inference worker thread failed: {e}")
            self.after(0, self._sync_inference_complete, f"⚠️ Error processing generation: {e}", None, None, None)

    def _sync_inference_complete(self, response: str, matrix, tokens, seq_ids):
        """Updates components inside main thread once calculations complete."""
        self.chat_panel.hide_typing_indicator()
        self.chat_panel.add_message("Bot", response)
        
        if matrix is not None and tokens is not None:
            # 1. Update live-canvas attention heatmap
            self.visual_panel.update_attention_heatmap(matrix, tokens)
            
            # 2. Update tokenization table grid
            self.visual_panel.clear_tokens_table()
            for idx, (word, tid) in enumerate(zip(tokens, seq_ids)):
                # Determine description types
                if word == "<PAD>":
                    w_type = "Special Padding Token"
                elif word == "<SEP>":
                    w_type = "Special Utterance Separator"
                elif word == "<EOS>":
                    w_type = "Special End-of-Sentence Token"
                elif word == "<UNK>":
                    w_type = "Unknown Out-Of-Vocabulary Word"
                else:
                    w_type = "Standard Vocabulary Word"
                    
                self.visual_panel.add_token_row(word, tid, w_type)

    # ==========================================
    # THREAD-SAFE TRAINING PIPELINES
    # ==========================================
    def action_start_training(self):
        """Compiles background thread and kicks off training."""
        if self.training_thread is not None and self.training_thread.is_alive():
            return
            
        # Clear panel chart
        self.training_panel.reset_chart()
        self.training_panel.set_training_state(True)
        self.safe_update_status("training")
        
        # Instantiate worker thread
        self.training_thread = TrainingWorker(self)
        self.training_thread.start()
        print("[*] Interface: Training worker thread successfully launched.")

    def action_stop_training(self):
        """Sends cancel signal to thread."""
        if self.training_thread is not None:
            self.training_thread.stop()

    def on_epoch_complete(self, epoch: int, max_epochs: int, loss: float):
        """Callback from training thread: updates chart and progress bar."""
        self.training_panel.update_training_progress(epoch, max_epochs, loss)

    def on_training_finished(self, success: bool):
        """Called when training finishes successfully or is stopped early."""
        self.training_panel.set_training_state(False)
        self.training_thread = None
        
        if success:
            print("[+] Interface: Training finished successfully!")
            self.action_reload_model()  # Reload active pipeline
        else:
            print("[-] Interface: Training was canceled early by user.")
            self.safe_update_status("offline")

    def on_training_failed(self, err_msg: str):
        """Called if background worker crashes."""
        self.training_panel.set_training_state(False)
        self.training_thread = None
        self.safe_update_status("offline")
        ctk.CTkMessagebox(
            title="Training Failure",
            message=f"An error occurred during model training: {err_msg}",
            icon="cancel"
        )

    def safe_update_status(self, status: str):
        """Updates the status display widgets safely."""
        self.current_status = status
        self.sidebar.update_model_info(
            device="CPU/GPU",
            vocab_size="N/A",
            layers=2,
            embed_dim=32,
            status=status
        )


if __name__ == "__main__":
    app = TransformerChatbotApp()
    app.mainloop()
