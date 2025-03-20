# SparseMamba-PCL

#### [📌] We have modified key parameters, and we will release the full details once the paper is accepted!

This repository is the official implementation of the paper SparseMamba-PCL: Scribble-Supervised Medical Image Segmentation via SAM-Guided Progressive Collaborative Learning [Arxiv](https://arxiv.org/abs/2503.01633).

If you have any questions about the code, please contact LUYI002@e.ntu.edu.sg.

## Datasets

### ACDC
1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
2. The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 
3. The pre-processed ACDC data used for training could be directly downloaded from [ACDC_dataset](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).

### CHAOs
1. The CHAOs dataset with mask annotations can be downloaded from [CHAOs](https://www.sciencedirect.com/science/article/abs/pii/S1361841520303145).
2. The scribble annotations of CHAOs, generated in [CHAOs scribbles](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_23). 

### MSCMRSeg
1. The MSCMRSeg dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html). 
2. The scribble annotations of MSCMRseg have been released in [MSCMRSeg_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMRSeg_scribbles). 
3. The scribble-annotated MSCMRSeg dataset used for training could be directly downloaded from [MSCMR_dataset](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_dataset).

## Requirements

Download the pretrained [Med-SAM](https://github.com/bowang-lab/MedSAM?tab=readme-ov-file) model [sam_vit_b_01ec64.pth](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth), and place it at e.g., `./SparseMamba-PCL`.

Some important required packages include:
* Python 3.8
* CUDA 11.8
* causal-conv1d 1.2.0.post2
* einops 0.8.0
* imageio 2.35.1
* mamba-ssm 2.2.2
* matplotlib 3.7.5
* medpy 0.5.2
* nibabel 5.2.1
* opencv-python 4.11.0.86
* pillow 10.2.0
* scikit-image 0.21.0
* scipy 1.10.1
* simpleitk 2.4.1
* six 1.17.0
* tensorboardx 2.6.2.2 
* torch 2.0.0+cu118
* torchaudio 2.0.1+cu118
* torchvision 0.15.1+cu118
* tqdm 4.67.1 

## Training

To train the model, run this command:

### train SparseMamba

```python train_weakly_supervised_SparseMamba.py```

### train SparseMamba_PCL

```python train_weakly_supervised_SparseMamba_PCL.py```


## Evaluation

To evaluate the model, run this command:

``` python test_2D_fully.py ```

## Citation

```bash
@article{qiu2025sparsemambapclscribblesupervisedmedicalimage,
      title={SparseMamba-PCL: Scribble-Supervised Medical Image Segmentation via SAM-Guided Progressive Collaborative Learning}, 
      author={Luyi Qiu and Tristan Till and Xiaobao Guo and Adams Wai-Kin Kong},
      year={2025},
      eprint={2503.01633},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.01633}
}
``` 
