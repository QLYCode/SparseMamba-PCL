


import warnings
warnings.filterwarnings(action="ignore")

from CycleMix.train_api import train

if __name__ == "__main__":
    train(output_dir="./checkpoints/PuzzleMix", box=False, graph=True)





