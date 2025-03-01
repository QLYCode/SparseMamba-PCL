import argparse
from random import random
import numpy as np
from torch.backends import cudnn
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        from networks.unet import UNet, UNet_DS, UNet_CCT_3H
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        from networks.unet_cct import UNet_CCT
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_3h":
        from networks.unet import UNet, UNet_DS, UNet_CCT_3H
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        from networks.unet import UNet, UNet_DS, UNet_CCT_3H
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        from networks.efficientunet import Effi_UNet
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        from networks.pnet import PNet2D
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()

    # ********************* CNN ***************************
    elif net_type == "ghostnet":
        from networks.ghostnet import ghostnet
        net = ghostnet().cuda()
    elif net_type == "ccnet":
        from networks.ccnet import CCNet_Model
        net = CCNet_Model(num_classes=class_num).cuda()
    elif net_type == "alignseg":
        from networks.alignseg import AlignSeg
        net = AlignSeg(num_classes=class_num).cuda()
    elif net_type == "bisenet":
        from networks.bisenet import BiSeNet
        net = BiSeNet(in_channel=in_chns, nclass=class_num, backbone='resnet18').to(device)
    elif net_type == "mobilenet":
        from networks.mobilenet import MobileNetV2
        net = MobileNetV2(input_channel=in_chns, n_class=class_num).cuda()
    elif net_type == "ecanet":
        from networks.ecanet import ECA_MobileNetV2
        net = ECA_MobileNetV2(n_class=4, width_mult=1).to(device)

    # ********************* transformer ***************************
    elif net_type == "segmenter":
        from networks.segmenter import create_segmenter
        #创建一个Segmenter类的实例

        model_cfg = {
            "image_size": (256, 256),
            "patch_size": 16,
            "d_model": 192,
            "n_heads": 3,
            "n_layers": 12,
            "normalization": "vit",
            "distilled": False,
            "backbone": "vit_tiny_patch16_384",
            "decoder": {
                "name": "mask_transformer",
                "drop_path_rate": 0.0,
                "dropout": 0.1,
                "n_layers": 2,
            },
            "n_cls": class_num,  # number of classes
        }
        net = create_segmenter(model_cfg).cuda()
    elif net_type == "unetformer":
        from networks.UNetFormer import UNetFormer
        net = UNetFormer(num_classes= class_num).cuda()
    elif net_type == "segformer":
        from networks.segformer import SegFormer
        net = SegFormer(num_classes= class_num).cuda()
    elif net_type == "swin_transformer":
        from networks.swin_transformer import SwinTransformer
        net = SwinTransformer().cuda()
    elif net_type == "swinunet":
        from networks.swin_unet import get_swin_unet
        net = get_swin_unet().cuda()
    elif net_type == "transunet":
        import argparse
        from networks.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        parser = argparse.ArgumentParser()
        parser.add_argument('--root_path', type=str,
                            default='../data/Synapse/train_npz', help='root dir for data')
        parser.add_argument('--dataset', type=str,
                            default='ACDC', help='experiment_name')
        parser.add_argument('--list_dir', type=str,
                            default='./lists/lists_Synapse', help='list dir')
        parser.add_argument('--num_classes', type=int,
                            default=4, help='output channel of network')
        parser.add_argument('--max_iterations', type=int,
                            default=30000, help='maximum epoch number to train')
        parser.add_argument('--max_epochs', type=int,
                            default=150, help='maximum epoch number to train')
        parser.add_argument('--batch_size', type=int,
                            default=24, help='batch_size per gpu')
        parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
        parser.add_argument('--deterministic', type=int, default=1,
                            help='whether use deterministic training')
        parser.add_argument('--base_lr', type=float, default=0.01,
                            help='segmentation network learning rate')
        parser.add_argument('--img_size', type=int,
                            default=224, help='input patch size of network input')
        parser.add_argument('--seed', type=int,
                            default=1234, help='random seed')
        parser.add_argument('--n_skip', type=int,
                            default=3, help='using number of skip-connect, default is num')
        parser.add_argument('--vit_name', type=str,
                            default='R50-ViT-B_16', help='select one vit model')
        parser.add_argument('--vit_patches_size', type=int,
                            default=16, help='vit_patches_size, default is 16')
        args = parser.parse_args()

        if not args.deterministic:
            cudnn.benchmark = True
            cudnn.deterministic = False
        else:
            cudnn.benchmark = False
            cudnn.deterministic = True

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        dataset_name = args.dataset

        args.num_classes = 4
        args.root_path = "/home/luyi/WSL4MIS/data/ACDC"
        args.list_dir = "/home/luyi/WSL4MIS/data/train.list"
        args.is_pretrain = True
        args.exp = 'transunet' + dataset_name + str(args.img_size)

        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()



    # ********************* mamba ***************************
    elif net_type == "msvmunet":
        from networks.msvmunet import build_msvmunet_model
        net = build_msvmunet_model(in_channels=3, num_classes=class_num).cuda()
    elif net_type == "segmamba":
        from networks.segmamba import SegMamba
        net = SegMamba(in_chns, class_num).cuda()
    elif net_type == "swinumamba":
        from networks.SwinUMamba import SwinUMamba
        net = SwinUMamba(in_chns, class_num).cuda()
    elif net_type == "emnet":
        from networks.em_net_model import EMNet
        net = EMNet(in_chns, class_num).cuda()
    elif net_type == "umamba":
        from networks.UMambaBot_2d import UMambaBot
        net = UMambaBot(input_channels=in_chns,
                        n_stages=4,  # 包含stem的4个编码阶段
                        features_per_stage=[32, 64, 128, 256],  # 各阶段特征通道数
                        conv_op=nn.Conv2d,
                        kernel_sizes=[[3,3], [3,3], [3,3], [3,3]],  # 每个阶段的卷积核尺寸
                        strides=[1, 2, 2, 2],  # 下采样策略（第一次保持分辨率）
                        n_conv_per_stage=[2, 2, 2, 2],  # 编码器各阶段卷积层数
                        num_classes=class_num,
                        n_conv_per_stage_decoder=[2, 2, 2],  # 解码器各阶段卷积层数（比编码器少1阶）
                        conv_bias=False,
                        norm_op=nn.BatchNorm2d,
                        norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
                        nonlin=nn.LeakyReLU,
                        nonlin_kwargs={"inplace": True},
                        deep_supervision=False
                        ).cuda()
    elif net_type == "lmunet":
        from networks.lkmunet import LKMUNet
        net = LKMUNet(in_channels=in_chns, out_channels=class_num, kernel_sizes=[21, 15, 9, 3]).cuda()

    # ********************* proposed ***************************
    elif net_type == "umamba_es2d":
        from networks.UMamba_e2sd import UMambaBot
        net = UMambaBot(input_channels=in_chns,
                        n_stages=4,  # 包含stem的4个编码阶段
                        features_per_stage=[32, 64, 128, 256],  # 各阶段特征通道数
                        conv_op=nn.Conv2d,
                        kernel_sizes=[[3,3], [3,3], [3,3], [3,3]],  # 每个阶段的卷积核尺寸
                        strides=[1, 2, 2, 2],  # 下采样策略（第一次保持分辨率）
                        n_conv_per_stage=[2, 2, 2, 2],  # 编码器各阶段卷积层数
                        num_classes=class_num,
                        n_conv_per_stage_decoder=[2, 2, 2],  # 解码器各阶段卷积层数（比编码器少1阶）
                        conv_bias=False,
                        norm_op=nn.BatchNorm2d,
                        norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
                        nonlin=nn.LeakyReLU,
                        nonlin_kwargs={"inplace": True},
                        deep_supervision=False
                        ).cuda()

    else:
        net = None
    return net

