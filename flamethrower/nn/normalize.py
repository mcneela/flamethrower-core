"""Backward-compatible normalization imports.

BatchNorm previously had a second, divergent implementation in this module.
Keeping this alias preserves existing course examples while ensuring both import
paths use the same parameters, running statistics, and train/eval behavior.
"""

from .batch_norm import BatchNorm

__all__ = ['BatchNorm']
