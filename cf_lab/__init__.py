"""Catastrophic Forgetting Lab.

An end-to-end experiment platform for studying catastrophic forgetting in
transformer language models (LLMs) under sequential fine-tuning.
"""

from .config import ExperimentConfig, detect_device, default_config

__version__ = "0.1.0"
__all__ = ["ExperimentConfig", "detect_device", "default_config", "__version__"]
