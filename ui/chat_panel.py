import customtkinter as ctk
import time
import styles as st

class ChatPanel(ctk.CTkFrame):
    """
    Central Conversation Viewport for the Transformer Chatbot.
    
    Features:
    - Scrollable messaging viewport (CTkScrollableFrame).
    - Speech bubbles styled differently for User (right, brand purple) and Bot (left, slate).
    - Multi-line automatic word wrapping.
    - Timestamp indicators on all bubble entries.
    - Dynamic, animated-style "Typing Indicator" bubble.
    - Input bar with auto-focus entry and custom SVG-style text send action button.
    """
    def __init__(self, parent, on_send_callback):
        super().__init__(parent, fg_color="transparent")
        
        self.on_send_callback = on_send_callback
        self.message_widgets = []
        self.typing_indicator_widget = None
        
        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Message scrollable area takes full height
        
        # 1. HEADER BAR
        self.header = ctk.CTkFrame(self, fg_color=st.BG_PANEL, height=60, corner_radius=0)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_columnconfigure(0, weight=1)
        
        self.header_title = ctk.CTkLabel(
            self.header,
            text="💬 NEURAL CHAT INTERFACE",
            font=st.FONT_SUBTITLE,
            text_color=st.ACCENT_PRIMARY
        )
        self.header_title.grid(row=0, column=0, padx=24, pady=15, sticky="w")
        
        # 2. SCROLLABLE CONVERSATION VIEWPORT
        self.chat_viewport = ctk.CTkScrollableFrame(
            self,
            fg_color=st.BG_MAIN,
            corner_radius=0
        )
        self.chat_viewport.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        self.chat_viewport.grid_columnconfigure(0, weight=1)
        
        # 3. INPUT CONTROL PANEL (Bottom Bar)
        self.input_container = ctk.CTkFrame(self, fg_color=st.BG_PANEL, height=80, corner_radius=0)
        self.input_container.grid(row=2, column=0, sticky="ew")
        self.input_container.grid_columnconfigure(0, weight=1)
        self.input_container.grid_rowconfigure(0, weight=1)
        
        # Text entry box
        self.entry = ctk.CTkEntry(
            self.input_container,
            placeholder_text="Enter neural prompt...",
            font=st.FONT_BODY,
            fg_color=st.BG_INPUT,
            border_color=st.ACCENT_SECONDARY,
            border_width=1,
            text_color=st.TEXT_MAIN,
            placeholder_text_color=st.TEXT_MUTED,
            corner_radius=st.CORNER_RADIUS_MD
        )
        self.entry.grid(row=0, column=0, padx=(24, 12), pady=20, sticky="ew")
        self.entry.bind("<Return>", lambda event: self.send_message())  # Press Enter key to send
        
        # Send button
        self.send_btn = ctk.CTkButton(
            self.input_container,
            text="GENERATE ⚡",
            font=st.FONT_HEADER,
            width=100,
            fg_color=st.ACCENT_PRIMARY_CONTAINER,
            hover_color=st.ACCENT_HOVER,
            text_color=st.TEXT_MAIN,
            corner_radius=st.CORNER_RADIUS_MD,
            command=self.send_message
        )
        self.send_btn.grid(row=0, column=1, padx=(0, 24), pady=20, sticky="ns")
        
        # Display initial greeting
        self.add_message("Bot", "Interface initialized. Neural network online. How can I assist your inquiry today?")

    def send_message(self):
        """Extracts text from input and triggers callback."""
        text = self.entry.get().strip()
        if not text:
            return
            
        # Clear input box
        self.entry.delete(0, "end")
        
        # Display user message bubble
        self.add_message("User", text)
        
        # Trigger parent callback to process through transformer models
        self.on_send_callback(text)

    def add_message(self, sender: str, text: str):
        """Creates and appends a beautifully padded alternating speech bubble."""
        # Main row container
        row_frame = ctk.CTkFrame(self.chat_viewport, fg_color="transparent")
        row_frame.grid(row=len(self.message_widgets), column=0, padx=20, pady=10, sticky="ew")
        row_frame.grid_columnconfigure(0, weight=1)
        
        # Setup sender styling
        timestamp = time.strftime("%H:%M:%S")
        
        if sender == "User":
            bubble_bg = st.BUBBLE_USER_BG
            bubble_border = 0
            border_color = "transparent"
            text_color = st.BUBBLE_USER_FG
            sticky_side = "e"  # Align to right
            pad_left = 80      # Reserve gap on left side
            pad_right = 0
        else:
            bubble_bg = st.BUBBLE_BOT_BG
            bubble_border = 1
            border_color = st.BUBBLE_BOT_BORDER
            text_color = st.BUBBLE_BOT_FG
            sticky_side = "w"  # Align to left
            pad_left = 0
            pad_right = 80     # Reserve gap on right side

        # Speech bubble card
        bubble_card = ctk.CTkFrame(
            row_frame, 
            fg_color=bubble_bg, 
            border_width=bubble_border,
            border_color=border_color,
            corner_radius=st.CORNER_RADIUS_LG
        )
        bubble_card.grid(row=0, column=0, padx=(pad_left, pad_right), sticky=sticky_side)
        
        # Text label (forces wrapping for long prompts)
        lbl = ctk.CTkLabel(
            bubble_card,
            text=text,
            font=st.FONT_BODY,
            text_color=text_color,
            justify="left",
            wraplength=450  # Wraps long lines to next line gracefully
        )
        lbl.grid(row=0, column=0, padx=18, pady=(12, 6), sticky="w")
        
        # Small timestamp display
        t_lbl = ctk.CTkLabel(
            bubble_card,
            text=f"[{sender.upper()} :: {timestamp}]",
            font=st.FONT_CODE,
            text_color=st.TEXT_MUTED if sender == "Bot" else "#d8b4fe"
        )
        t_lbl.grid(row=1, column=0, padx=18, pady=(0, 8), sticky="e")
        
        self.message_widgets.append(row_frame)
        self.update_viewport_scroll()

    def show_typing_indicator(self):
        """Creates a temporary animated bubble to represent chatbot processing."""
        if self.typing_indicator_widget is not None:
            return
            
        self.typing_indicator_widget = ctk.CTkFrame(self.chat_viewport, fg_color="transparent")
        self.typing_indicator_widget.grid(row=len(self.message_widgets) + 1, column=0, padx=20, pady=10, sticky="ew")
        
        bubble = ctk.CTkFrame(
            self.typing_indicator_widget, 
            fg_color=st.BUBBLE_BOT_BG, 
            border_width=1,
            border_color=st.BUBBLE_BOT_BORDER,
            corner_radius=st.CORNER_RADIUS_LG
        )
        bubble.grid(row=0, column=0, padx=(0, 80), sticky="w")
        
        lbl = ctk.CTkLabel(
            bubble,
            text="SYNTHESIZING RESPONSE...",
            font=st.FONT_CODE,
            text_color=st.ACCENT_SECONDARY
        )
        lbl.grid(row=0, column=0, padx=20, pady=15)
        
        self.update_viewport_scroll()

    def hide_typing_indicator(self):
        """Destroys and removes the thinking bubble."""
        if self.typing_indicator_widget is not None:
            self.typing_indicator_widget.destroy()
            self.typing_indicator_widget = None

    def update_viewport_scroll(self):
        """Forces the scrollbar to bottom to view new messages."""
        self.chat_viewport.update_idletasks()
        # CustomTkinter Scrollable Frame uses vertical scrollbar updates
        self.chat_viewport._parent_canvas.yview_moveto(1.0)

    def clear_chat(self):
        """Wipes conversation visualization from panel."""
        for widget in self.message_widgets:
            widget.destroy()
        self.message_widgets = []
        self.hide_typing_indicator()
        self.add_message("Bot", "Conversation history cleared. Let's start fresh!")
