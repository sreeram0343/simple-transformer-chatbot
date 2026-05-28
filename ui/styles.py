# Styles and UI Design Tokens for the Transformer Chatbot Dashboard

# Color Palette (Dark Theme / Catppuccin and Claude-inspired Slate Theme)
BG_MAIN = "#1e1e2e"          # Main window background (deep dark violet-slate)
BG_PANEL = "#252538"         # Sidebars and card backgrounds
BG_CARD = "#2c2c44"          # Widget card containers and panels
BG_INPUT = "#181825"         # Input entries background
TEXT_MAIN = "#f8f9fa"        # Primary white text
TEXT_MUTED = "#a9afb8"       # Secondary muted gray text
ACCENT_PRIMARY = "#7b2cbf"   # Brand primary (deep purple)
ACCENT_LIGHT = "#9d4edd"     # Brand highlight (light purple)
ACCENT_HOVER = "#5a189a"     # Brand primary hover color

# Chat Bubbles Design
BUBBLE_USER_BG = "#7b2cbf"   # User bubble background (purple)
BUBBLE_USER_FG = "#ffffff"   # User bubble text color
BUBBLE_BOT_BG = "#32324d"    # Bot bubble background (slate gray)
BUBBLE_BOT_FG = "#f8f9fa"    # Bot bubble text color

# Alert / Status Colors
COLOR_SUCCESS = "#2ec4b6"    # Green (Ready status)
COLOR_WARNING = "#ffb703"    # Orange/Yellow (Training status)
COLOR_ERROR = "#e63946"      # Red (Offline/Error status)

# Typography Configuration
FONT_FAMILY = "Segoe UI"
FONT_CODE_FAMILY = "Consolas"

# Font Sizes & Weights
FONT_TITLE = (FONT_FAMILY, 18, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 14, "bold")
FONT_HEADER = (FONT_FAMILY, 12, "bold")
FONT_BODY = (FONT_FAMILY, 11, "normal")
FONT_MUTED_SMALL = (FONT_FAMILY, 9, "normal")
FONT_CODE = (FONT_CODE_FAMILY, 10, "normal")

# Component Dimensions & Rounding
CORNER_RADIUS_SM = 6         # Small elements (buttons, small status panels)
CORNER_RADIUS_MD = 12        # Medium elements (speech bubbles, entry fields)
CORNER_RADIUS_LG = 16        # Large panels (cards, sidebars, main frames)
