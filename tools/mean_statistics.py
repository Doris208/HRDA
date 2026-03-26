# calc_remote_sensing_stats.py
# 用于快速计算遥感数据集 (IRRG, RGB 等) 通道均值 mean 和方差 std
# 把这个脚本放在服务器上运行。

import cv2
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 设置你想计算的数据集的图片根目录
IMG_DIR = '/root/shared-nvme/ISPRS_dataset/Potsdam/RGB/img_dir/train' # 比如 Vaihingen 的训练图片目录

# 2. 你的图片有多少个通道？(遥感通常是 3 或 4)
NUM_CHANNELS = 3 

# 3. 如果是 3 通道，你算 mean 的目的是为了填入 mmseg 的 dict(to_rgb=True) 还是 to_rgb=False?
# 这决定了算出来的 mean 的顺序。
# 如果你像 GLGAN 包装的那样，IRRG 直接就是通道顺序 (IR->R->G)，不需要BGR转RGB，建议选 False。
# 如果你要适配传统的 to_rgb=True (BGR->RGB)，则计算时需要反转 BGR。
# 【强烈建议遥感选 False】，保持物理通道顺序。
CONVERT_TO_RGB = True 

# ===========================================

def calculate_stats():
    img_list = [os.path.join(IMG_DIR, x) for x in os.walk(IMG_DIR).__next__()[2] if x.endswith('.png') or x.endswith('.tif')]
    print(f"Found {len(img_list)} images. Starting calculation...")

    channels_sum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    channels_squared_sum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    pixels_count = 0

    for img_path in tqdm(img_list):
        # 1. 读取图片
        # 注意：cv2.imread 默认读取 PNG/TIF 为 BGR 通道顺序，如果是单通道或4通道需要另外处理
        # 针对 3 通道遥感 PNG：
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image: {img_path}")
                continue
        except Exception as e:
            print(f"Exception loading image: {img_path}, error: {e}")
            continue

        h, w, c = img.shape
        if c != NUM_CHANNELS:
            print(f"Warning: Image {img_path} has {c} channels, expected {NUM_CHANNELS}. Skipping.")
            continue

        # 2. 处理通道顺序
        if CONVERT_TO_RGB:
            # 标准 BGR -> RGB 反转
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 遥感的 BGR 含义通常是固定的：
        # 如果你做 Vaihingen->Potsdam UDA
        # GLGAN (P1) 里的物理顺序：Vaihingen 是 IR-R-G (对应 B-G-R); Potsdam 是 R-G-B (对应 B-G-R)
        # 所以你不需要转换，cv2 读出来的 B-G-R 顺序就是 IR-R-G。

        img = img.astype(np.float64)

        # 3. 累加均值和平方均值
        for i in range(NUM_CHANNELS):
            channels_sum[i] += np.sum(img[:, :, i])
            channels_squared_sum[i] += np.sum(img[:, :, i]**2)
        
        pixels_count += h * w

    # 4. 计算最终的均值和方差
    final_mean = channels_sum / pixels_count
    
    # 方差 Var = E(X^2) - (E(X))^2
    final_var = (channels_squared_sum / pixels_count) - (final_mean**2)
    final_std = np.sqrt(final_var)

    print("\n" + "="*40)
    print(f"Dataset Stats for: {IMG_DIR}")
    print("-" * 40)
    
    # 按照mmseg规范输出数值：to_rgb=False 保持 BGR 顺序。
    # 如果 Vaihingen，物理通道是 IR-R-G。
    print(f"Final Mean (BGR/IRRG order, to_rgb={CONVERT_TO_RGB}):")
    print([f"{x:.3f}" for x in final_mean])
    
    print("-" * 40)
    print(f"Final Std (BGR/IRRG order, to_rgb={CONVERT_TO_RGB}):")
    print([f"{x:.3f}" for x in final_std])
    print("="*40)
    print("Now update this in your dataset config file!")

if __name__ == "__main__":
    calculate_stats()