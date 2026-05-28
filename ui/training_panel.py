import customtkinter as ctk
import matplotlib
# Use the TkAgg backend specifically for embedding matplotlib inside tkinter widgets
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import styles as st

class TrainingPanel(ctk.CTkFrame):
    """
    Real-Time Training Dashboard for the Transformer Chatbot.
    
    Features:
    - Embedding of a live-updating Matplotlib canvas displaying the training loss curve.
    - Start and Stop training controller buttons.
    - Status progress bar displaying epochs complete.
    - Live data dashboard cards showing Epoch and Loss statistics.
    - Background statistics card describing dataset parameters.
    """
    def __init__(self, parent, on_start_training, on_stop_training):
        super().__init__(parent, fg_color="transparent")
        
        self.on_start_training = on_start_training
        self.on_stop_training = on_stop_training
        
        # Loss plot historical arrays
        self.epoch_history = []
        self.loss_history = []
        
        # Configure layout (2 columns: left is dashboard & specs, right is graph)
        self.grid_columnconfigure(0, weight=1)  # Controls and Info
        self.grid_columnconfigure(1, weight=1)  # Loss Plot canvas
        self.grid_rowconfigure(0, weight=1)
        
        # ==========================================
        # LEFT COLUMN: METRICS & CONTROLS
        # ==========================================
        self.left_container = ctk.CTkFrame(self, fg_color="transparent")
        self.left_container.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        self.left_container.grid_columnconfigure(0, weight=1)
        
        # 1. Action Controls Panel
        self.controls_card = ctk.CTkFrame(self.left_container, fg_color=st.BG_CARD, corner_radius=st.CORNER_RADIUS_LG)
        self.controls_card.grid(row=0, column=0, padx=0, pady=10, sticky="ew")
        self.controls_card.grid_columnconfigure(0, weight=1)
        self.controls_card.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            self.controls_card,
            text="TRAINING CONTROLLER",
            font=st.FONT_HEADER,
            text_color=st.TEXT_MUTED
        ).grid(row=0, column=0, columnspan=2, padx=20, pady=(18, 10), sticky="w")
        
        self.start_btn = ctk.CTkButton(
            self.controls_card,
            text="▶ Start Training",
            font=st.FONT_HEADER,
            fg_color=st.COLOR_SUCCESS,
            hover_color="#22998f",
            text_color="#181825",
            corner_radius=st.CORNER_RADIUS_SM,
            command=self.on_start_training
        )
        self.start_btn.grid(row=1, column=0, padx=(20, 5), pady=(5, 10), sticky="ew")
        
        self.stop_btn = ctk.CTkButton(
            self.controls_card,
            text="⏹ Stop Training",
            font=st.FONT_HEADER,
            fg_color=st.COLOR_ERROR,
            hover_color="#b52c38",
            text_color="#ffffff",
            state="disabled",  # Start out disabled
            corner_radius=st.CORNER_RADIUS_SM,
            command=self.on_stop_training
        )
        self.stop_btn.grid(row=1, column=1, padx=(5, 20), pady=(5, 10), sticky="ew")
        
        # Progress Bar
        self.progress_label = ctk.CTkLabel(
            self.controls_card,
            text="Training progress: 0%",
            font=st.FONT_BODY,
            text_color=st.TEXT_MUTED
        )
        self.progress_label.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 2), sticky="w")
        
        self.progress_bar = ctk.CTkProgressBar(
            self.controls_card,
            fg_color=st.BG_INPUT,
            progress_color=st.ACCENT_PRIMARY,
            corner_radius=st.CORNER_RADIUS_SM
        )
        self.progress_bar.grid(row=3, column=0, columnspan=2, padx=20, pady=(2, 20), sticky="ew")
        self.progress_bar.set(0)
        
        # 2. Real-Time Dashboard Stats Card
        self.stats_card = ctk.CTkFrame(self.left_container, fg_color=st.BG_CARD, corner_radius=st.CORNER_RADIUS_LG)
        self.stats_card.grid(row=1, column=0, padx=0, pady=10, sticky="ew")
        self.stats_card.grid_columnconfigure(0, weight=1)
        self.stats_card.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            self.stats_card,
            text="REAL-TIME METRICS",
            font=st.FONT_HEADER,
            text_color=st.TEXT_MUTED
        ).grid(row=0, column=0, columnspan=2, padx=20, pady=(18, 10), sticky="w")
        
        # Epoch Tracker Grid Card
        self.epoch_box = ctk.CTkFrame(self.stats_card, fg_color=st.BG_INPUT, corner_radius=st.CORNER_RADIUS_MD)
        self.epoch_box.grid(row=1, column=0, padx=(20, 10), pady=(5, 20), sticky="nsew")
        self.epoch_label = ctk.CTkLabel(self.epoch_box, text="Epoch", font=st.FONT_MUTED_SMALL, text_color=st.TEXT_MUTED)
        self.epoch_label.pack(padx=10, pady=(10, 0))
        self.epoch_val = ctk.CTkLabel(self.epoch_box, text="0 / 40", font=(st.FONT_FAMILY, 20, "bold"), text_color=st.TEXT_MAIN)
        self.epoch_val.pack(padx=10, pady=(2, 10))
        
        # Loss Tracker Grid Card
        self.loss_box = ctk.CTkFrame(self.stats_card, fg_color=st.BG_INPUT, corner_radius=st.CORNER_RADIUS_MD)
        self.loss_box.grid(row=1, column=1, padx=(10, 20), pady=(5, 20), sticky="nsew")
        self.loss_label = ctk.CTkLabel(self.loss_box, text="Loss", font=st.FONT_MUTED_SMALL, text_color=st.TEXT_MUTED)
        self.loss_label.pack(padx=10, pady=(10, 0))
        self.loss_val = ctk.CTkLabel(self.loss_box, text="0.0000", font=(st.FONT_FAMILY, 20, "bold"), text_color=st.TEXT_MAIN)
        self.loss_val.pack(padx=10, pady=(2, 10))
        
        # 3. Dataset Info Card
        self.dataset_card = ctk.CTkFrame(self.left_container, fg_color=st.BG_CARD, corner_radius=st.CORNER_RADIUS_LG)
        self.dataset_card.grid(row=2, column=0, padx=0, pady=10, sticky="ew")
        self.dataset_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            self.dataset_card,
            text="DATASET STATISTICS",
            font=st.FONT_HEADER,
            text_color=st.TEXT_MUTED
        ).grid(row=0, column=0, padx=20, pady=(15, 8), sticky="w")
        
        self._create_stats_row(self.dataset_card, "Dataset File:", "data/conversations.txt", 1)
        self._create_stats_row(self.dataset_card, "Conversation Pairs:", "10 turns", 2)
        self._create_stats_row(self.dataset_card, "Sequence Truncation:", "32 tokens", 3)
        ctk.CTkLabel(self.dataset_card, text="", height=8).grid(row=4, column=0)
        
        # ==========================================
        # RIGHT COLUMN: GRAPH CANVAS EMBED
        # ==========================================
        self.right_container = ctk.CTkFrame(self, fg_color=st.BG_CARD, corner_radius=st.CORNER_RADIUS_LG)
        self.right_container.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        self.right_container.grid_columnconfigure(0, weight=1)
        self.right_container.grid_rowconfigure(1, weight=1)
        
        # Plot Header
        self.plot_title = ctk.CTkLabel(
            self.right_container,
            text="📈 Loss Reduction History",
            font=st.FONT_SUBTITLE,
            text_color=st.TEXT_MAIN
        )
        self.plot_title.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")
        
        # Matplotlib Frame Container
        self.plot_frame = ctk.CTkFrame(self.right_container, fg_color="transparent")
        self.plot_frame.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="nsew")
        
        # Setup Figure and Subplot
        # We specify background colors to integrate cleanly with CustomTkinter styles
        self.fig, self.ax = plt.subplots(figsize=(5, 3.8), facecolor=st.BG_CARD)
        self.ax.set_facecolor(st.BG_INPUT)
        
        # Configure initial chart styling
        self.ax.set_title("Training Loss over Epochs", color=st.TEXT_MAIN, fontsize=10, pad=10)
        self.ax.set_xlabel("Epoch", color=st.TEXT_MUTED, fontsize=8)
        self.ax.set_ylabel("Loss", color=st.TEXT_MUTED, fontsize=8)
        self.ax.tick_params(colors=st.TEXT_MUTED, labelsize=8)
        
        # Hide standard border lines (spines) for modern look
        for spine in self.ax.spines.values():
            spine.set_color(st.BG_CARD)
            
        self.ax.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
        
        # Embed Figure into Tkinter Widget via FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _create_stats_row(self, parent, label_text, val_text, row_idx):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row_idx, column=0, padx=20, pady=4, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        lbl = ctk.CTkLabel(frame, text=label_text, font=st.FONT_BODY, text_color=st.TEXT_MUTED)
        lbl.grid(row=0, column=0, sticky="w")
        
        val = ctk.CTkLabel(frame, text=val_text, font=st.FONT_BODY, text_color=st.TEXT_MAIN)
        val.grid(row=0, column=1, sticky="e")

    def update_training_progress(self, current_epoch: int, max_epochs: int, loss: float):
        """Called by background thread via main window to safely update GUI elements."""
        self.epoch_val.configure(text=f"{current_epoch} / {max_epochs}")
        self.loss_val.configure(text=f"{loss:.4f}")
        
        # Set progress bar
        pct = current_epoch / max_epochs
        self.progress_bar.set(pct)
        self.progress_label.configure(text=f"Training progress: {int(pct * 100)}%")
        
        # Append data to histories
        self.epoch_history.append(current_epoch)
        self.loss_history.append(loss)
        
        # Redraw the loss plot in place
        self.ax.clear()
        self.ax.set_facecolor(st.BG_INPUT)
        self.ax.set_title("Training Loss over Epochs", color=st.TEXT_MAIN, fontsize=10, pad=10)
        self.ax.set_xlabel("Epoch", color=st.TEXT_MUTED, fontsize=8)
        self.ax.set_ylabel("Loss", color=st.TEXT_MUTED, fontsize=8)
        self.ax.tick_params(colors=st.TEXT_MUTED, labelsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
        for spine in self.ax.spines.values():
            spine.set_color(st.BG_CARD)
            
        # Draw the historical data curve
        self.ax.plot(
            self.epoch_history, 
            self.loss_history, 
            color=st.ACCENT_LIGHT, 
            marker="o", 
            markersize=4,
            linewidth=2, 
            label="Cross Entropy Loss"
        )
        self.ax.legend(facecolor=st.BG_CARD, edgecolor="none", labelcolor=st.TEXT_MAIN, fontsize=8)
        
        # Redraw canvas
        self.canvas.draw()

    def set_training_state(self, is_training: bool):
        """Enables/Disables controls appropriately to prevent double running."""
        if is_training:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def reset_chart(self):
        """Clears chart datasets."""
        self.epoch_history = []
        self.loss_history = []
        self.ax.clear()
        self.ax.set_facecolor(st.BG_INPUT)
        self.ax.set_title("Training Loss over Epochs", color=st.TEXT_MAIN, fontsize=10, pad=10)
        self.ax.set_xlabel("Epoch", color=st.TEXT_MUTED, fontsize=8)
        self.ax.set_ylabel("Loss", color=st.TEXT_MUTED, fontsize=8)
        self.ax.tick_params(colors=st.TEXT_MUTED, labelsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
        for spine in self.ax.spines.values():
            spine.set_color(st.BG_CARD)
        self.canvas.draw()
        self.progress_bar.set(0)
        self.progress_label.configure(text="Training progress: 0%")
        self.epoch_val.configure(text="0 / 40")
        self.loss_val.configure(text="0.0000")
