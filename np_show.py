import numpy as np
import matplotlib.pyplot as plt

# 假设img是一个已经加载好的numpy数组，代表图像
# 例如：img = np.random.rand(100, 100)  # 随机生成一个100x100的图像

img=np.load("/home/yuezk/yzk/SegSEU/Lite/MedSAM/test_demo/litemedsam-seg/2DBox_CXR_demo.npz")

# 显示图像
plt.imshow(img)  # 'gray'是色彩映射，适合灰度图，彩色图像可以去掉这个参数
# plt.colorbar()  # 显示色标
plt.show()
