# 教师和学上网络都输入1024*1024图像 需要将repvit_sam
import os
import random
import monai
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from glob import glob
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import cv2
import torch.nn.functional as F

from segment_anything.build_sam import build_sam_vit_b
from matplotlib import pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=str, default="./data/npy",
    help="Path to the npy data root."
)
parser.add_argument(
    "-pretrained_teacher", type=str, default="./work_dir/medsam_vit_b.pth",
    help="Path to the pretrained MedSAM checkpoint."
)
parser.add_argument(
    "-resume", type=str, default='workdir/latest_student_model.pth',
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=str, default="./workdir",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=5,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=1,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=0,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training.边界框扰动"
)
parser.add_argument(
    "-lr", type=float, default=0.00005,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)
parser.add_argument(
    "-iou_loss_weight", type=float, default=1.0,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.0,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)
args=parser.parse_args()
args = parser.parse_args()
# %%
work_dir = args.work_dir
data_root = args.data_root
teacher_checkpoint = args.pretrained_teacher
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
device = args.device
bbox_shift = args.bbox_shift
lr = args.lr
weight_decay = args.weight_decay
iou_loss_weight = args.iou_loss_weight
seg_loss_weight = args.seg_loss_weight
ce_loss_weight = args.ce_loss_weight
do_sancheck = args.sanity_check
checkpoint = args.resume
makedirs(work_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

def cal_iou(result, reference):
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])

    iou = intersection.float() / union.float()

    return iou.unsqueeze(1)

"""定义数据集"""

class NpyDataset(Dataset):
    def __init__(self, data_root, image_size=1024,  data_aug=False):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
        self.gt_path_files = [
            file for file in self.gt_path_files
            if isfile(join(self.img_path, basename(file)))
        ]
        self.image_size = image_size
        self.target_length = image_size
        # self.bbox_shift = bbox_shift
        self.data_aug = data_aug

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + \
                                                                self.npy_files[index]
        img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True)  # (H, W, 3)
        img_resize = self.resize_longest_side(img_3c)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8,
                                                               a_max=None)  # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize)  # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1))  # (3, 256, 256)
        assert np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True)  # multiple labels [0, 1,4,5...], (256,256)
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = self.pad_image(gt)  # (256, 256)
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))  # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt))  # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        # if self.data_aug:
        #     if random.random() > 0.5:
        #         img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
        #         gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
        #         # print('DA with flip left right')
        #     if random.random() > 0.5:
        #         img_padded = np.ascontiguousarray(
        #             np.flip(img_padded, axis=-2))  # 先沿着倒数第二个轴对 img_padded 进行翻转操作，然后将结果转换为 C 连续数组。
        #         gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        #         # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)
        # y_indices, x_indices = np.where(gt2D > 0)
        # x_min, x_max = np.min(x_indices), np.max(x_indices)
        # y_min, y_max = np.min(y_indices), np.max(y_indices)
        # # add perturbation to bounding box coordinates
        # H, W = gt2D.shape
        # x_min = max(0, x_min - random.randint(0, self.bbox_shift))  # 对边界框进行随机扰动，以增加数据的多样性。
        # x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        # y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        # y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        # bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :, :]).long(),
            # "bboxes": torch.tensor(bboxes[None, None, ...]).float(),  # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    def resize_longest_side(self, image):  # 其最长的边达到指定的长度 self.target_length。

        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):  # 填充图片到指定H,W
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3:  ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else:  ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded

#%% sanity test of dataset class
if do_sancheck:
    tr_dataset = NpyDataset(data_root, data_aug=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, batch in enumerate(tr_dataloader):
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        image = batch["image"]
        # gt = batch["gt2D"]
        # bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
        # show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        # show_box(bboxes[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
        # show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        # show_box(bboxes[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        # set title
        axs[1].set_title(names_temp[idx])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break

"""dataloader"""
train_dataset = NpyDataset(data_root=data_root, data_aug=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

"""构建学生模型"""
from repvit_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from timm.models import create_model
medsam_lite_image_encoder = create_model('repvit')
class Student_model(nn.Module):
    def __init__(self,image_encoder):
        super(Student_model,self).__init__()
        self.image_encoder=image_encoder
    def forward(self,x):
        return self.image_encoder(x)
student_model=Student_model(medsam_lite_image_encoder)
student_model.to(device)
student_model.train()


"""教师模型,将lite_sam作为微调模型"""
#以medsam作为教师模型
# sam_model = build_sam_vit_b(teacher_checkpoint)
# sam_image_encoder = sam_model.image_encoder
# sam_image_encoder_state_dict = sam_image_encoder.state_dict()
# class Teacher_model(nn.Module):
#     def __init__(self,sam_image_encoder):
#         super(Teacher_model,self).__init__()
#         self.image_encoder = sam_image_encoder
#     def forward(self,x):
#         #输出相同向量
#         return self.image_encoder(x)
# teacher_model=Teacher_model(sam_image_encoder)
# teacher_model.image_encoder.load_state_dict(sam_image_encoder_state_dict, True)  #采用GPU加载的话,seganything/build_sam.py的144行的device改成cuda：0
# teacher_model.to(device)
# teacher_model.eval()

medsam_model = sam_model_registry["vit_b"](checkpoint=args.pretrained_teacher)
medsam_model = medsam_model.to(device)
medsam_model.eval()

"""优化器、损失函数"""
T=7  #蒸馏温度
optimizer = optim.AdamW(
    student_model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
)
soft_loss=nn.KLDivLoss(reduction="batchmean").to(device)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.9,
#     patience=5,
#     cooldown=0
# )
#，当验证损失不再下降时，学习率会逐渐减小，这通常有助于模型在训练后期达到更好的性能
# train_loss = train(...)
# # Validation code
# val_loss = validate(...)
# # Update the learning rate
# lr_scheduler.step(val_loss)
# class SoftTargetCrossEntropy(nn.Module):
#
#     def __init__(self):
#         super(SoftTargetCrossEntropy, self).__init__()
#
#     def forward(self, x, target):
#         loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
#         return loss.mean()
# criterion = SoftTargetCrossEntropy()

"""恢复中断"""
if checkpoint and isfile(checkpoint):
    print(f"Resuming from checkpoint {checkpoint}")
    checkpoint = torch.load(checkpoint)
    student_model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    start_epoch = 0
    best_loss = 1e10

"""加载数据"""
"""dataloader"""
train_dataset = NpyDataset(data_root=data_root, data_aug=True)
print(f"Dataset length: {len(train_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
"""开始训练"""
train_losses = []
for epoch in range(start_epoch , num_epochs):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        image = batch["image"]
        image = image.to(device)
        with torch.no_grad():
            # print(f"Input shape: {image.shape}")

            teacher_output = medsam_model.image_encoder(image)
            # print(f"Teacher output shape: {teacher_output.shape}")
        optimizer.zero_grad()
        output=student_model(image)
        loss = soft_loss(F.log_softmax(output/T,dim=1),F.softmax(teacher_output.detach()/T,dim=1))
        epoch_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(
            f"Epoch {epoch+1} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")
    epoch_end_time = time()
    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    train_losses.append(epoch_loss_reduced)
    model_weights = student_model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss,
    }
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    torch.save(checkpoint, join(work_dir, f"student_model_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.pth"))
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
    best_loss = epoch_loss_reduced
    checkpoint["best_loss"] = best_loss
    torch.save(checkpoint, join(work_dir, "medsam_lite_best.pth"))
    epoch_loss_reduced = 1e10
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(join(work_dir, "train_loss.png"))
plt.close()







"""还需完成
1、增加标签的loss：loss = a* CE(l_stu, gt)+ b * KLD(l_stu, l_tea)
2、图像增强
3、先保存教师的logits：1、采用同种数据增强方法：采用相同的seed（参考tinyvit）
                    2、加载logits，计算loss"""


