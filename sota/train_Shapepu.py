
import warnings
warnings.filterwarnings(action="ignore")

from ShapePU.train_api import train


if __name__ == "__main__":
    train(dataset="ACDC_dataset", output_dir="./checkpoints/Shapepu", epochs=200)
    # train(dataset="MSCMR_dataset", output_dir="./checkpoints/Shapepu", epochs=200)



