#!/usr/bin/env python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from src.vqvae.training.main import main

if __name__ == "__main__":
    main()
