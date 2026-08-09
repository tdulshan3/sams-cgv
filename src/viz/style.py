"""M9 Visualisation & QA — House style.

Consistent fonts, a colour-blind-safe palette, no chartjunk.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

from src.config import CHART_STYLE, CHART_FIGSIZE

# Colour-blind safe palette
# Green/Red alone is bad for colour-blindness.
# Adding distinct lightness and a third category (uncertain) helps.
# In charts, we will also use position/shape where possible.
COLOURS = {
    "present": "#2E7D32",   # Dark green
    "absent": "#C62828",    # Dark red
    "uncertain": "#F9A825"  # Yellow/Orange
}

def apply_house_style() -> None:
    """Apply the standard house style to all matplotlib charts."""
    plt.style.use(CHART_STYLE)
    
    # Customise further for no chartjunk and better readability
    mpl.rcParams["figure.figsize"] = CHART_FIGSIZE
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10
    mpl.rcParams["legend.frameon"] = False
    
    # Remove top and right spines to reduce chartjunk
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
