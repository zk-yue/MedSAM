import torch
import torch.nn as nn
import time
# -*- coding: utf-8 -*-
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse



@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="assets/img_demo.png",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="assets/",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--box",
    type=list,
    default=[47, 35, 146, 219],
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
args = parser.parse_args()

device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

img_np = np.load("./data/sample/npy/imgs/XRay_COVID-19-Radiography-Database_COVID-1.npy")
img_gt=np.load("./data/sample/npy/gts/XRay_COVID-19-Radiography-Database_COVID-1.npy")
img_np=img_np[:,:,1]
# img_gt=img_gt[:,:,2]
# img_np = np.load("./data/sample/npy/imgs/XRay_Chest-Xray-Masks-and-Labels_MCUCXR_0393_1.npy")
# img_gt=np.load("./data/sample/npy/gts/XRay_Chest-Xray-Masks-and-Labels_MCUCXR_0393_1.npy")

# img_np = io.imread(args.data_path)
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape
# %% image preprocessing
img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

box_np = np.array([args.box])
# transfer box_np t0 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024

with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
io.imsave(
    join(args.seg_path, "seg_medsam" + os.path.basename(args.data_path)),
    medsam_seg,
    check_contrast=False,
)

# 导入repvit编码器
from repvit_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from timm.models import create_model
from torch import nn
class Student_model(nn.Module):
    def __init__(self,image_encoder):
        super(Student_model,self).__init__()
        self.image_encoder=image_encoder
    def forward(self,x):
        return self.image_encoder(x)
repvit_image_encoder = Student_model(create_model('repvit'))
checkpoint="./work_dir/RepViT_MedSAM/student_model_2024_05_08_23_42.pth"
with open(checkpoint, "rb") as f:
    state_dict = torch.load(f)
repvit_image_encoder.load_state_dict(state_dict['model'])
medsam_model.image_encoder=repvit_image_encoder
medsam_model = medsam_model.to(device)
medsam_model.eval()

with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

repvit_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
io.imsave(
    join(args.seg_path, "seg_repvit" + os.path.basename(args.data_path)),
    repvit_seg,
    check_contrast=False,
)





















# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # 假设输入图像大小为32x32

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        return x

# 创建模型实例并转移到适当的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 创建一个随机输入张量，假设输入大小为32x32 RGB图像
input_tensor = torch.randn(1, 3, 32, 32, device=device)

# 热身GPU（如果使用的话），进行几次前向传播以避免启动延迟
for _ in range(10):
    model(input_tensor)

# 开始计时
start_time = time.time()

# 执行正向推理
output = model(input_tensor)

# 结束计时
end_time = time.time()

# 计算并打印所需时间
print(f"Forward pass took {end_time - start_time:.6f} seconds.")