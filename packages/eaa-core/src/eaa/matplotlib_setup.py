"""Configure matplotlib for headless image generation."""

import matplotlib


matplotlib.use("Agg", force=True)
