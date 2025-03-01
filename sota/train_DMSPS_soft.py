
import warnings
warnings.filterwarnings(action="ignore")


from DMSPS.code.train.A_train_weaklySup_SPS_2d_soft import train_api

if __name__ == "__main__":
    train_api(root_path="/home/linux/Desktop/WSL4MIS/seg00/DMSPS/data/ACDC2017/ACDC_for2D",
              outputdir="./checkpoints/DMSPS-soft")



