"""
拼图脚本
"""


import os
from PIL import Image

# ================= 配置区域 =================

# 1. 输入和输出路径
SOURCE_DIR = r"/Users/mengzijie/Downloads/technical_report/k4.2.4/output"
FINAL_IMAGE_PATH = os.path.join(SOURCE_DIR, "Figure_Stylized_Subjects_v2.png")

# 2. 【关键修改】每一行单独指定要选取的帧文件名(不带后缀)
# 格式 -> '文件夹名': [图片名1, 图片名2, ...]
FRAME_CONFIG = {
    '1': [3, 6, 16, 68, 118, 131],        # 第1个视频选第1,2,3,4,5,6秒
    '4': [1, 406, 61, 83, 437, 445],       # 第2个视频选第5到10秒
    '3': [3, 39, 74, 17, 141,10],       # 第3个视频每隔一秒选一张 # 266
    '2': [1, 5, 147, 216, 41 ,14],       # 第4个视频... #420
    '5': [1, 21, 32, 131, 40, 36],        # ...
    '6': [1, 10, 19, 40, 138, 600],   # ...
    # '7': [2, 14, 71, 223]
}

# 3. 样式设置
TARGET_WIDTH = 500  # 单张小图的统一宽度(px)
H_SPACING = 0       # 横向间距
V_SPACING = 15      # 纵向间距

# ============================================

def create_figure():
    print(f"开始拼图，目标宽度: {TARGET_WIDTH}px")
    
    # 获取有序的文件夹列表 ('1', '2', ... '6')
    # 这里我们直接用 FRAME_CONFIG 的 key 进行排序，确保按你配置的顺序画图
    sorted_folders = sorted(FRAME_CONFIG.keys(), key=lambda x: int(x))
    sorted_folders = list(FRAME_CONFIG.keys())
    
    row_images = []

    # --- 逐行处理 ---
    for folder_name in sorted_folders:
        folder_path = os.path.join(SOURCE_DIR, folder_name)
        selected_indices = FRAME_CONFIG[folder_name]
        
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_name} 不存在，跳过。")
            continue

        print(f"正在处理文件夹 {folder_name}，选取帧: {selected_indices}")

        images_in_row = []
        
        for frame_idx in selected_indices:
            img_name = f"{frame_idx}.png"
            img_path = os.path.join(folder_path, img_name)
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                
                # 等比缩放
                aspect_ratio = img.height / img.width
                new_height = int(TARGET_WIDTH * aspect_ratio)
                img_resized = img.resize((TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)
                
                images_in_row.append(img_resized)
            else:
                print(f"  错误: 图片 {img_name} 在文件夹 {folder_name} 中找不到，使用空白填充。")
                # 找不到图时用白色方块占位，防止报错
                images_in_row.append(Image.new('RGB', (TARGET_WIDTH, int(TARGET_WIDTH * 0.75)), (255, 255, 255)))

        if not images_in_row:
            continue

        # 拼接当前行
        # 计算行总宽
        row_width = sum(img.width for img in images_in_row) + (len(images_in_row) - 1) * H_SPACING
        # 行总高 (取最大值)
        row_height = max(img.height for img in images_in_row)
        
        row_canvas = Image.new('RGB', (row_width, row_height), (255, 255, 255))
        
        current_x = 0
        for img in images_in_row:
            row_canvas.paste(img, (current_x, 0))
            current_x += img.width + H_SPACING
            
        row_images.append(row_canvas)

    if not row_images:
        print("未生成任何图像。")
        return

    # --- 纵向堆叠所有行 ---
    final_width = max(row.width for row in row_images)
    final_height = sum(row.height for row in row_images) + (len(row_images) - 1) * V_SPACING
    
    final_canvas = Image.new('RGB', (final_width, final_height), (255, 255, 255))
    
    current_y = 0
    for row in row_images:
        final_canvas.paste(row, (0, current_y))
        current_y += row.height + V_SPACING
        
    final_canvas.save(FINAL_IMAGE_PATH, quality=95)
    print(f"\n拼图完成！已保存至: {FINAL_IMAGE_PATH}")

if __name__ == "__main__":
    create_figure()