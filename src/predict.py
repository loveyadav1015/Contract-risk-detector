"""Inference logic.

This module will handle:
- Loading a trained model checkpoint
- Running inference on new contract text
- Returning predicted risk level with confidence score
"""

import torch
from transformers import AutoTokenizer
