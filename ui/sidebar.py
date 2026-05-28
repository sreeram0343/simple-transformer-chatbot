import customtkinter as ctk
import styles as st

class SidebarPanel(ctk.CTkFrame):
    """
    Left-docked Sidebar Panel for the Transformer Chatbot GUI.
    
    Features:
    - Branding logo / Title.
    - Active status indicator (Ready, Training, Offline).
    - Model metadata display (Device, Layers, Embedding Dimension, Vocab Size).
    - Session-wide control buttons (Reset Memory, Reload Model).
    """
    def __init__(self, parent, on_clear_callback, on_reload_callback):
        super().__init__(parent, fg_color=st.BG_PANEL, corner_radius=0)
        
        self.on_clear_callback = on_clear_callback
        self.on_reload_callback = on_reload_callback
        
        # Configure layout (single-column grid)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)  # Spacer row to push controls to bottom
        
        # 1. BRANDING / LOGO HEADER
        self.logo_label = ctk.CTkLabel(
            self, 
            text="🔮 Mini Bot", 
            font=st.FONT_TITLE,
            text_color=st.TEXT_MAIN
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5), sticky="w")
        
        self.version_label = ctk.CTkLabel(
            self,
            text="Transformer v1.0 • Scratch",
            font=st.FONT_MUTED_SMALL,
            text_color=st.TEXT_MUTED
        )
        self.version_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")
        
        # 2. MODEL STATUS INDICATOR CARD
        self.status_card = ctk.CTkFrame(self, fg_color=st.BG_CARD, corner_radius=st.CORNER_RADIUS_MD)
        self.status_card.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        self.status_card.grid_columnconfigure(1, weight=1)
        
        # Colored circle representing state
        self.status_dot = ctk.CTkFrame(
            self.status_card, 
            width=12, 
            height=12, 
            corner_radius=6, 
            fg_color=st.COLOR_ERROR
        )
        self.status_dot.grid(row=0, column=0, padx=(15, 10), pady=15)
        
        self.status_text = ctk.CTkLabel(
            self.status_card,
            text="Offline (No Model)",
            font=st.FONT_HEADER,
            text_color=st.TEXT_MAIN
        )
        self.status_text.grid(row=0, column=1, padx=(0, 15), pady=15, sticky="w")
        
        # 3. METADATA INFO PANEL (STATS CARD)
        self.stats_card = ctk.CTkFrame(self, fg_color=st.BG_CARD, corner_radius=st.CORNER_RADIUS_MD)
        self.stats_card.grid(row=3, column=0, padx=15, pady=10, sticky="ew")
        self.stats_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            self.stats_card, 
            text="MODEL SPECIFICATIONS", 
            font=st.FONT_HEADER,
            text_color=st.TEXT_MUTED
        ).grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        
        # Info Fields
        self.info_device = self._create_info_row(self.stats_card, "Active Device:", "Detecting...", 1)
        self.info_vocab = self._create_info_row(self.stats_card, "Vocab Size:", "N/A", 2)
        self.info_layers = self._create_info_row(self.stats_card, "Stacked Blocks:", "2 Layers", 3)
        self.info_embed = self._create_info_row(self.stats_card, "Embed Dim (d_model):", "32 Dimensions", 4)
        
        # Bottom spacer inside stats card
        ctk.CTkLabel(self.stats_card, text="", height=5).grid(row=5, column=0)
        
        # 4. ACTION BUTTONS PANEL (Bottom of Sidebar)
        self.actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.actions_frame.grid(row=6, column=0, padx=15, pady=25, sticky="ew")
        self.actions_frame.grid_columnconfigure(0, weight=1)
        
        # Refresh Model weights
        self.reload_btn = ctk.CTkButton(
            self.actions_frame,
            text="🔄 Reload Model",
            font=st.FONT_HEADER,
            fg_color=st.BG_CARD,
            hover_color=st.BG_INPUT,
            text_color=st.TEXT_MAIN,
            corner_radius=st.CORNER_RADIUS_SM,
            command=self.on_reload_callback
        )
        self.reload_btn.grid(row=0, column=0, padx=0, pady=5, sticky="ew")
        
        # Clear chat memory buffer
        self.clear_btn = ctk.CTkButton(
            self.actions_frame,
            text="🧹 Reset Memory",
            font=st.FONT_HEADER,
            fg_color=st.ACCENT_PRIMARY,
            hover_color=st.ACCENT_HOVER,
            text_color=st.TEXT_MAIN,
            corner_radius=st.CORNER_RADIUS_SM,
            command=self.on_clear_callback
        )
        self.clear_btn.grid(row=1, column=0, padx=0, pady=5, sticky="ew")

    def _create_info_row(self, parent, label_text, val_text, row_idx):
        """Helper to create a nice-looking key-value metadata row."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row_idx, column=0, padx=15, pady=4, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        lbl = ctk.CTkLabel(frame, text=label_text, font=st.FONT_BODY, text_color=st.TEXT_MUTED)
        lbl.grid(row=0, column=0, sticky="w")
        
        val = ctk.CTkLabel(frame, text=val_text, font=st.FONT_HEADER, text_color=st.TEXT_MAIN)
        val.grid(row=0, column=1, sticky="e")
        return val

    def update_model_info(self, device: str, vocab_size: int, layers: int, embed_dim: int, status: str):
        """Updates the sidebar values dynamically based on loaded model state."""
        self.info_device.configure(text=str(device).upper())
        self.info_vocab.configure(text=str(vocab_size))
        self.info_layers.configure(text=f"{layers} Blocks")
        self.info_embed.configure(text=f"{embed_dim} Dim")
        
        # Update status indicator widget
        if status == "ready":
            self.status_dot.configure(fg_color=st.COLOR_SUCCESS)
            self.status_text.configure(text="Ready (Idle)", text_color=st.TEXT_MAIN)
        elif status == "training":
            self.status_dot.configure(fg_color=st.COLOR_WARNING)
            self.status_text.configure(text="Training model...", text_color=st.COLOR_WARNING)
        else:
            self.status_dot.configure(fg_color=st.COLOR_ERROR)
            self.status_text.configure(text="Offline (No Model)", text_color=st.TEXT_MUTED)
