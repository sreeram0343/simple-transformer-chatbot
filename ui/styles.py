# Styles and UI Design Tokens for the Transformer Chatbot Dashboard
# Updated to "Neural Cyber-Noir" Design System

# Color Palette (Cyber-Noir Theme)
BG_MAIN = "#121221"          # Main background (surface)
BG_PANEL = "#1e1e2e"         # Surface container (sidebar/header)
BG_CARD = "#1a1a2a"          # Surface container low
BG_INPUT = "#0d0d1c"         # Surface container lowest
TEXT_MAIN = "#e3e0f7"        # On-surface
TEXT_MUTED = "#cfc2d5"       # On-surface-variant

# Accents & Branding
ACCENT_PRIMARY = "#deb7ff"   # Electric Indigo (Primary)
ACCENT_PRIMARY_CONTAINER = "#7b2cbf" 
ACCENT_SECONDARY = "#4fdbcc" # Neon Teal (Secondary)
ACCENT_SECONDARY_CONTAINER = "#00b1a3"
ACCENT_HOVER = "#8234c6"     # Inverse primary

# Chat Bubbles Design
BUBBLE_USER_BG = "#7b2cbf"   # Electric Indigo to Deep Violet gradient base
BUBBLE_USER_FG = "#ffffff"
BUBBLE_BOT_BG = "#1e1e2e"    # Frosted glass look base
BUBBLE_BOT_FG = "#e3e0f7"
BUBBLE_BOT_BORDER = "#4fdbcc" # Neon Teal border

# Alert / Status Colors
COLOR_SUCCESS = "#4fdbcc"    # Neon Teal (Online)
COLOR_WARNING = "#deb7ff"    # Electric Indigo (Processing)
COLOR_ERROR = "#ffb4ab"      # Error

# Typography Configuration
# Prefer Geist and JetBrains Mono if available, fallback to system defaults
FONT_FAMILY = "Geist"
FONT_CODE_FAMILY = "JetBrains Mono"

# Font Sizes & Weights
FONT_TITLE = (FONT_FAMILY, 24, "bold")      # Headline-md
FONT_SUBTITLE = (FONT_FAMILY, 18, "bold")   # Headline-sm
FONT_HEADER = (FONT_FAMILY, 14, "bold")     # Body-md bold
FONT_BODY = (FONT_FAMILY, 12, "normal")     # Body-md
FONT_MUTED_SMALL = (FONT_FAMILY, 10, "normal")
FONT_CODE = (FONT_CODE_FAMILY, 11, "normal") # Data-display

# Component Dimensions & Rounding
CORNER_RADIUS_SM = 4         # rounded-sm
CORNER_RADIUS_MD = 8         # rounded-lg
CORNER_RADIUS_LG = 12        # rounded-xl
