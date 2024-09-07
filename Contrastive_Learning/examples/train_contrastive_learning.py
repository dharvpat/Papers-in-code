import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.training.train_contrastive import train_contrastive_learning

if __name__ == "__main__":
    train_contrastive_learning()