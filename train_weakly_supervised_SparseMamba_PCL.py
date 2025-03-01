import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
# from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, MSCMRDataset, ChaosDataset
# import networks.MedSAM_Inference
from networks.MedSAM_Inference import medsam_model, medsam_inference
from networks.net_factory import net_factory
from utils.spobe import ObjectPseudoBoundaryGenerator
from val_2D import test_single_volume,test_single_volume_ds
from utils import losses, metrics, ramps
import cv2
# from val_2D import test_single_volume, test_single_volume_ds
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
# ACDC      '../data/ACDC'
# chaos     '../data/CHAOs'
# MSCMR     '../data/MSCMR'
parser.add_argument('--exp', type=str,
                    default='ACDC_sparsemamba_pcl', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='fold5', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
# unet ccnet alignseg bisenet mobilenet ecanet efficientumamba segmamba umamba swinumamba lkmunet segmamba segmenter unetformer segformer sparsemamba
parser.add_argument('--model', type=str,
                    default='sparsemamba', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=120000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.03,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=2025, help='random seed')
parser.add_argument('--threshold', type=float, default=0.4,
                    help='threshold for edge detection')
parser.add_argument('--kernel_sizes', type=int, default=7, 
                    help='kernel size for edge detection')
args = parser.parse_args()

def aug_label(label_batch, volume_batch, boundary_generator):
    pseudo_edges = boundary_generator(volume_batch, label_batch)
    label_batch = torch.where(pseudo_edges > 0, label_batch, label_batch)
    return label_batch

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    W,H = args.patch_size
    if args.root_path == '../data/ACDC':
        db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]), fold=args.fold, sup_type=args.sup_type)
        db_val = BaseDataSets(base_dir=args.root_path,
                              fold=args.fold, split="val")
    elif args.root_path == '../data/CHAOs':
        db_train = ChaosDataset(args.root_path, split="train", transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]))
        db_val = ChaosDataset(args.root_path, split="val")
    elif args.root_path == '../data/MSCMR':
        db_train = MSCMRDataset(args.root_path, split="train", transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]))
        db_val = MSCMRDataset(args.root_path, split="val")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            """
            # SPOBE Generator
            boundary_generator = ObjectPseudoBoundaryGenerator(
                k=25,
                kernel_sizes=[7, 13, 25],
                sobel_threshold=args.threshold,
                ignore_index=4,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Enriched Scibble with boundary generator
            label_batch = aug_label(label_batch, volume_batch, boundary_generator)
            """


            volume_batch = volume_batch.cuda()
            label_batch = label_batch.cuda()

            # 一次性转换并插值所有图像，避免单张处理
            volume_batch_3c = volume_batch.repeat(1, 3, 1, 1)  # (B,3,256,256)
            volume_batch_3c_1024 = F.interpolate(volume_batch_3c, size=(1024, 1024),
                                                 mode="bilinear", align_corners=False)
            with torch.no_grad():
                # 整个 batch 一次性编码
                batch_image_embedding = medsam_model.image_encoder(volume_batch_3c_1024)

            outputs = model(volume_batch)  # generate coarse prediction
            outputs_soft = torch.softmax(outputs, dim=1)
            try:
                #将中间层的特征图进行卷积，与image_embedding进行相加
                middle = model.atten_out
                middle = model.conv_sam(middle)
                batch_image_embedding = torch.add(batch_image_embedding, middle)
            except:
                pass

            # 提取 coarse prediction 的边界框和边缘检测
            outputs_np = torch.argmax(outputs_soft, dim=1).cpu().numpy()  # Shape: (B, H, W)
            label_np = label_batch.cpu().numpy()  # Shape: (B, H, W)
            bounding_boxes = []
            edges_list = []

            medsam_seg_list = []
            for class_id in range(num_classes):  # 遍历每个类别
                medsam_seg_batch_list = []
                for batch_idx in range(outputs_np.shape[0]):  # 遍历 batch 中的每个样本
                    class_mask = (outputs_np[batch_idx] == class_id).astype(np.uint8)  # 类别 mask
                    label_class_mask = (label_np[batch_idx] == class_id).astype(np.uint8)  # 真实标签的类别 mask

                    if class_mask.any() and label_class_mask.any():  # 如果类别 mask 不为空
                        # 提取 bounding box
                        import cv2
                        # cv2.Sobel
                        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        x, y, w, h = cv2.boundingRect(contours[0])
                        bounding_boxes.append([batch_idx, class_id, x, y, w, h])

                        # 边缘检测
                        edges = cv2.Canny(class_mask * 255, 100, 200)
                        edges_list.append((batch_idx, class_id, edges))

                        # 记录到 TensorBoard
                        box_np = np.array([x, y, x + w, y + h])
                        box_1024 = box_np / np.array([W, H, W, H]) * 1024  # 转换到 1024x1024 尺度

                        #提取涂鸦标注的bounding box
                        contours, _ = cv2.findContours(label_class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        x_scr, y_scr, w_scr, h_scr = cv2.boundingRect(contours[0])
                        edges_scr = cv2.Canny(label_class_mask * 255, 100, 200)
                        box_scr_np = np.array([x_scr, y_scr, x_scr + w_scr, y_scr + h_scr])
                        box_scr_1024 = box_scr_np / np.array([W, H, W, H]) * 1024  # 转换到 1024x1024 尺度

                        #将box_scr_1024向四周扩展指定像素
                        ext_pixels = 10   #这里的像素数可以视作一个超参数，可以自行调整
                        box_scr_1024_Aug = box_scr_1024.copy()
                        box_scr_1024_Aug[0] = max(0, box_scr_1024_Aug[0] - ext_pixels)
                        box_scr_1024_Aug[1] = max(0, box_scr_1024_Aug[1] - ext_pixels)
                        box_scr_1024_Aug[2] = min(1024, box_scr_1024_Aug[2] + ext_pixels)
                        box_scr_1024_Aug[3] = min(1024, box_scr_1024_Aug[3] + ext_pixels)

                        #最终的bounding box使用box_scr_1024_Aug和box_1024的交集
                        box_1024 = np.array([max(box_scr_1024_Aug[0], box_1024[0]), max(box_scr_1024_Aug[1], box_1024[1]),
                                                min(box_scr_1024_Aug[2], box_1024[2]), min(box_scr_1024_Aug[3], box_1024[3])])


                        # 使用 med-sam 生成分割结果
                        with torch.no_grad():
                            # 利用已经编码好的 batch_image_embedding[sample_idx] 进行推理
                            single_embed = batch_image_embedding[batch_idx].unsqueeze(0)
                            medsam_seg = medsam_inference(medsam_model, single_embed, box_1024.reshape(1, 4), H, W)
                        #由于mdsam_seg为二分类，标签0为背景，1为目标，但是我们的类别标签是0-3，所以需要将medsam_seg的标签0转换为4，1转换为class_id
                        medsam_seg[medsam_seg == 0] = 4
                        medsam_seg[medsam_seg == 1] = class_id

                        # 将结果记录到 TensorBoard
                        medsam_seg_tensor = torch.tensor(medsam_seg).unsqueeze(0).float().to(
                            label_batch.device)  # (1, H, W)
                        medsam_seg_batch_list.append(medsam_seg_tensor)
                        writer.add_image(f'train/medsam_seg_class_{class_id}_batch_{batch_idx}', medsam_seg_tensor,
                                         iter_num)

                        # print(f"Batch {batch_idx}, Class {class_id}: Bounding Box (1024x1024 scale): {box_1024}")
                    else:
                        # 如果类别 mask 为空，添加一个全零张量作为占位符
                        empty_seg_tensor = torch.zeros((1, H, W), dtype=torch.float).to(label_batch.device)
                        medsam_seg_batch_list.append(empty_seg_tensor)
                medsam_seg_list.append(medsam_seg_batch_list)
            # 记录边缘检测到 TensorBoard
            for batch_idx, class_id, edges in edges_list:
                edges_tensor = torch.tensor(edges).unsqueeze(0).float()  # (1, 1, H, W)
                writer.add_image(f'train/edges_class_{class_id}_batch_{batch_idx}', edges_tensor, iter_num)



            # Add medsam segmentation loss to the total loss
            medsam_seg_output = torch.cat(
                [torch.cat(medsam_seg_class_list, dim=0) for medsam_seg_class_list in medsam_seg_list], dim=0).view(
                label_batch.shape[0], num_classes, 256, 256).to(label_batch.device)


            #构造伪标签
            alpha = random.uniform(0.0, 1.0)
            # pseudo_label = alpha * medsam_seg_output + (1 - alpha) * outputs
            # 损失计算和优化

            medsam_seg_loss = ce_loss(medsam_seg_output, label_batch[:].long())
            pseudo_label = medsam_seg_output
            pseudo_label = torch.argmax(pseudo_label, dim=1)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            l_pls = 0.5 * (dice_loss(outputs, pseudo_label))


            # Total loss includes both coarse prediction and medsam_seg_loss
            #lambda是一个超参数，可以自行调整
            lambda_ = 0.3
            loss = loss_ce + medsam_seg_loss + 0.5 * l_pls
            # loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/medsam_seg_loss', medsam_seg_loss, iter_num)
            writer.add_scalar('info/l_pls', l_pls, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, medsam_seg_loss: %f,l_pls: %f' %
                (iter_num, loss.item(), loss_ce.item(), medsam_seg_loss.item(), l_pls.item())
            )
            # logging.info(
            #     'iteration %d : loss : %f, loss_ce: %f' %
            #     (iter_num, loss.item(), loss_ce.item())
            # )

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(outputs_soft, dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                        snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
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

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    def remove_readonly(func, path, excinfo):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)


    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code', onerror=remove_readonly)
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)