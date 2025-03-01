

import warnings
warnings.filterwarnings(action="ignore")


from ScribFormer.train_api import train


# # python train.py --root_path /home/linux/Desktop/WSL4MIS/data/ACDC --exp 111  --max_epoches 200 
if __name__ == "__main__":
    train(root_path="/home/linux/Desktop/WSL4MIS/data/ACDC", output_dir="./checkpoints", max_epoches=200)

