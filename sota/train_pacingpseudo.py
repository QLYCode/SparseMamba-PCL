

import warnings
warnings.filterwarnings(action="ignore")

from pacingpseudo.train_acdc import train_main

if __name__ == "__main__":
    train_main(output_dir="./checkpoints/pacingpseudo")



