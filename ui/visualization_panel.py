import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import styles as st

class VisualizationPanel(ctk.CTkFrame):
    """
    Visualization Panel displaying internal Transformer state representations.
    
    Contains 3 Interactive Sub-tabs:
    1. Tokenization Tab: Displays lists of input words mapped to integer Token IDs.
    2. Attention Heatmap Tab: Embeds a Matplotlib 2D heatmap showing the calculated self-attention
       matrix between sequence tokens in real-time.
    3. Architecture Tab: Formats a professional visual layout showing the structural elements.
    """
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        
        # Configure layout (single row/col carrying a CTkTabview)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 1. INITIALIZE TABVIEW
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=st.BG_CARD,
            segmented_button_fg_color=st.BG_MAIN,
            segmented_button_selected_color=st.ACCENT_PRIMARY,
            segmented_button_selected_hover_color=st.ACCENT_HOVER,
            segmented_button_unselected_color=st.BG_PANEL,
            text_color=st.TEXT_MAIN,
            corner_radius=st.CORNER_RADIUS_LG
        )
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Create separate tabs
        self.tab_tokens = self.tabview.add("🪙 Tokenization Grid")
        self.tab_attention = self.tabview.add("🧠 Self-Attention Matrix")
        self.tab_architecture = self.tabview.add("🧱 Model Architecture")
        
        # Configure grid for all tabs to maximize sizing
        for tab in [self.tab_tokens, self.tab_attention, self.tab_architecture]:
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)
            
        # 2. SETUP TOKENIZATION TAB
        self._setup_tokenization_tab()
        
        # 3. SETUP ATTENTION HEATMAP TAB
        self._setup_attention_tab()
        
        # 4. SETUP ARCHITECTURE TAB
        self._setup_architecture_tab()

    # ==========================================
    # TOKENIZATION TAB
    # ==========================================
    def _setup_tokenization_tab(self):
        frame = ctk.CTkFrame(self.tab_tokens, fg_color="transparent")
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        
        # Header Info
        ctk.CTkLabel(
            frame,
            text="🪙 Token-to-ID Numerical Vector Mapping",
            font=st.FONT_SUBTITLE,
            text_color=st.TEXT_MAIN
        ).grid(row=0, column=0, pady=(0, 5), sticky="w")
        
        ctk.CTkLabel(
            frame,
            text="Transformers read numbers, not words. The sentence is tokenized, mapped to unique IDs, and passed to the model.",
            font=st.FONT_BODY,
            text_color=st.TEXT_MUTED
        ).grid(row=1, column=0, pady=(0, 20), sticky="w")
        
        # Scrollable table list
        self.tokens_scroll = ctk.CTkScrollableFrame(frame, fg_color=st.BG_INPUT, corner_radius=st.CORNER_RADIUS_MD)
        self.tokens_scroll.grid(row=2, column=0, sticky="nsew")
        self.tokens_scroll.grid_columnconfigure(0, weight=1)
        self.tokens_scroll.grid_columnconfigure(1, weight=1)
        self.tokens_scroll.grid_columnconfigure(2, weight=1)
        
        # Table Header
        self._create_table_header(self.tokens_scroll, "Word Token", 0)
        self._create_table_header(self.tokens_scroll, "Token ID", 1)
        self._create_table_header(self.tokens_scroll, "Internal Meaning Type", 2)
        
        # Placeholder row
        self.token_rows = []
        self.add_token_row("hello", 4, "Regular Vocabulary Word")
        self.add_token_row("<SEP>", 2, "Special Turn Separator")
        self.add_token_row("hi", 5, "Regular Vocabulary Word")
        self.add_token_row("<EOS>", 3, "Special End-of-Sentence Token")

    def _create_table_header(self, parent, text, col):
        lbl = ctk.CTkLabel(
            parent,
            text=text,
            font=st.FONT_HEADER,
            text_color=st.ACCENT_LIGHT,
            pady=8
        )
        lbl.grid(row=0, column=col, sticky="nsew", padx=10)

    def add_token_row(self, word, token_id, token_type):
        """Adds a row to the token display table."""
        row_idx = len(self.token_rows) + 1
        
        # Word Token
        lbl_word = ctk.CTkLabel(self.tokens_scroll, text=word, font=st.FONT_CODE, text_color=st.TEXT_MAIN)
        lbl_word.grid(row=row_idx, column=0, padx=10, pady=4)
        
        # Token ID
        lbl_id = ctk.CTkLabel(self.tokens_scroll, text=str(token_id), font=st.FONT_CODE, text_color=st.TEXT_MAIN)
        lbl_id.grid(row=row_idx, column=1, padx=10, pady=4)
        
        # Token Type
        lbl_type = ctk.CTkLabel(self.tokens_scroll, text=token_type, font=st.FONT_BODY, text_color=st.TEXT_MUTED)
        lbl_type.grid(row=row_idx, column=2, padx=10, pady=4)
        
        self.token_rows.append((lbl_word, lbl_id, lbl_type))

    def clear_tokens_table(self):
        """Clears rows from token display."""
        for word, token_id, token_type in self.token_rows:
            word.destroy()
            token_id.destroy()
            token_type.destroy()
        self.token_rows = []

    # ==========================================
    # ATTENTION HEATMAP TAB
    # ==========================================
    def _setup_attention_tab(self):
        frame = ctk.CTkFrame(self.tab_attention, fg_color="transparent")
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)
        
        # Header Info
        ctk.CTkLabel(
            frame,
            text="🧠 Self-Attention Matrix Heatmap",
            font=st.FONT_SUBTITLE,
            text_color=st.TEXT_MAIN
        ).grid(row=0, column=0, pady=(0, 5), sticky="w")
        
        ctk.CTkLabel(
            frame,
            text="Displays attention scores between each pair of tokens. Darker spots indicate stronger attention weights (the model connects these words).",
            font=st.FONT_BODY,
            text_color=st.TEXT_MUTED
        ).grid(row=1, column=0, pady=(0, 10), sticky="w")
        
        # Matplotlib Area
        self.plot_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.plot_frame.grid(row=2, column=0, sticky="nsew")
        
        # Setup Figure and Subplot
        self.attn_fig, self.attn_ax = plt.subplots(figsize=(6, 4.5), facecolor=st.BG_CARD)
        self.attn_ax.set_facecolor(st.BG_INPUT)
        
        # Mock initial attention matrix for visualization on boot
        mock_tokens = ["hello", "<SEP>", "hi", "<EOS>"]
        mock_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0],
            [0.1, 0.1, 0.8, 0.0],
            [0.2, 0.1, 0.3, 0.4]
        ])
        
        self._plot_attention(mock_matrix, mock_tokens)
        
        # Embed Figure into Tkinter
        self.attn_canvas = FigureCanvasTkAgg(self.attn_fig, master=self.plot_frame)
        self.attn_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _plot_attention(self, attention_matrix, tokens):
        self.attn_ax.clear()
        
        # Plot 2D heatmap using imshow
        im = self.attn_ax.imshow(attention_matrix, cmap="Purples", vmin=0.0, vmax=1.0, aspect="auto")
        
        # Show all ticks and label them with the respective list entries
        self.attn_ax.set_xticks(np.arange(len(tokens)))
        self.attn_ax.set_yticks(np.arange(len(tokens)))
        self.attn_ax.set_xticklabels(tokens, color=st.TEXT_MAIN, fontname=st.FONT_FAMILY, fontsize=8, rotation=45, ha="right")
        self.attn_ax.set_yticklabels(tokens, color=st.TEXT_MAIN, fontname=st.FONT_FAMILY, fontsize=8)
        
        # Add labels and title
        self.attn_ax.set_title("Scaled Dot-Product Attention Heatmap (Head 0)", color=st.TEXT_MAIN, fontsize=10, pad=12)
        
        # Style spines
        for spine in self.attn_ax.spines.values():
            spine.set_color(st.BG_CARD)
            
        # Draw colorbar matching theme
        if not hasattr(self, 'colorbar') or self.colorbar is None:
            self.colorbar = self.attn_fig.colorbar(im, ax=self.attn_ax, fraction=0.046, pad=0.04)
            self.colorbar.ax.yaxis.set_tick_params(colors=st.TEXT_MUTED, labelsize=8)
            self.colorbar.outline.set_visible(False)
        else:
            self.colorbar.update_normal(im)
            
        self.attn_fig.tight_layout()

    def update_attention_heatmap(self, attention_matrix: np.ndarray, tokens: list[str]):
        """Called by app dynamically upon inference generation."""
        self._plot_attention(attention_matrix, tokens)
        self.attn_canvas.draw()

    # ==========================================
    # ARCHITECTURE SCHEMATIC TAB
    # ==========================================
    def _setup_architecture_tab(self):
        # We display a beautiful custom scrollable summary outlining the layer shapes
        scroll = ctk.CTkScrollableFrame(self.tab_architecture, fg_color="transparent")
        scroll.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            scroll,
            text="🧱 Custom GPT-Style Decoder-Only Architecture",
            font=st.FONT_SUBTITLE,
            text_color=st.TEXT_MAIN
        ).grid(row=0, column=0, pady=(0, 15), sticky="w")
        
        # 1. Inputs Block
        self._create_arch_block(
            scroll, "1. INPUT SEQUENCE & ENCODING", 
            "Input Text -> Token ID Sequence of shape: (batch_size, seq_len)\ne.g. 'hello how are you' -> [4, 18, 12, 16]\nPadding is added at the end up to max_len (32 tokens).", 
            1
        )
        
        # 2. Embeddings Block
        self._create_arch_block(
            scroll, "2. REPRESENTATION & POSITIONING", 
            "Token Embedding (d_model=32) + Sinusoidal Positional Encoding (d_model=32)\n- Adds continuous vectors to mapping IDs.\n- Injects positional coordinates using sine & cosine waveforms.\nResulting shape: (batch_size, seq_len, 32)", 
            2
        )
        
        # 3. Stacked Block
        self._create_arch_block(
            scroll, "3. STACKED TRANSFORMER BLOCKS (x2 Stacked)", 
            "For each block, the sequence passes through:\n\n"
            "↳ Pre-Layer Normalization 1\n"
            "↳ Multi-Head Attention (4 Parallel Heads, d_k = 32 / 4 = 8 Dimensions per head)\n"
            "  - Uses Causal Masking (upper-triangular -inf) to block looking at future tokens.\n"
            "↳ Residual Skip Addition (x = x + Attention(Norm(x)))\n\n"
            "↳ Pre-Layer Normalization 2\n"
            "↳ Position-Wise Feed-Forward Network (Linear Expansion d_ff=128 -> GELU -> Dropout -> Linear Projection back to 32)\n"
            "↳ Residual Skip Addition (x = x + FFN(Norm(x)))", 
            3
        )
        
        # 4. Heads Block
        self._create_arch_block(
            scroll, "4. FINAL NORMALIZATION & LANGUAGE MODEL HEAD", 
            "↳ Layer Normalization (Final)\n"
            "↳ Linear Classification Head (Weights tied directly to Token Embeddings)\n"
            "  - Projects 32 dimensions up to Vocab Size.\n"
            "Resulting Output Shape: (batch_size, seq_len, vocab_size) representing raw next-token probabilities (logits).", 
            4
        )

    def _create_arch_block(self, parent, title, body_text, row_idx):
        card = ctk.CTkFrame(parent, fg_color=st.BG_INPUT, border_color=st.BG_CARD, border_width=1, corner_radius=st.CORNER_RADIUS_MD)
        card.grid(row=row_idx, column=0, padx=0, pady=10, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            card,
            text=title,
            font=st.FONT_HEADER,
            text_color=st.ACCENT_LIGHT
        ).grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")
        
        ctk.CTkLabel(
            card,
            text=body_text,
            font=st.FONT_BODY,
            text_color=st.TEXT_MAIN,
            justify="left"
        ).grid(row=1, column=0, padx=20, pady=(5, 15), sticky="w")
