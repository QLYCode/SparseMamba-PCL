

import warnings
warnings.filterwarnings(action="ignore")


from ScribbleVC.train_api import train

if __name__ == "__main__":
    train(root_path="/home/linux/Desktop/WSL4MIS/data/ACDC", output_dir="./checkpoints", max_epoches=200)


