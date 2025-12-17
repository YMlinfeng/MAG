"""
拆图脚本
"""

import cv2
import os

# ================= 配置区域 =================

# 1. 输入：你的视频所在的文件夹路径
INPUT_DIR = r"/Users/mengzijie/Downloads/technical_report/1" 

# 2. 输出：你想把图片保存到哪里
OUTPUT_DIR = r"/Users/mengzijie/Downloads/technical_report/1/output"

# =======================================================

def process_videos():
    # 检查输入路径是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误：找不到输入文件夹 -> {INPUT_DIR}")
        return

    # 支持的视频格式后缀
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    # 获取所有视频文件并排序
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(video_extensions)]
    files.sort()

    if not files:
        print("未在目标文件夹中找到视频文件。")
        return

    print(f"共找到 {len(files)} 个视频，准备开始处理...")

    # 遍历视频
    for video_index, filename in enumerate(files, start=1):
        video_path = os.path.join(INPUT_DIR, filename)
        
        # ================= 修改点开始 =================
        # 1. 获取不带后缀的文件名 (例如: "data.mp4" -> "data")
        file_name_no_ext = os.path.splitext(filename)[0]
        
        # 2. 拼接新的文件夹名称： "编号_原始文件名" (例如: "1_data")
        new_folder_name = f"{video_index}_{file_name_no_ext}"
        
        # 3. 生成完整的输出路径
        current_output_folder = os.path.join(OUTPUT_DIR, new_folder_name)
        # ================= 修改点结束 =================

        os.makedirs(current_output_folder, exist_ok=True)

        print(f"正在处理第 {video_index} 个视频: {filename} -> 目标文件夹: {new_folder_name}")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  无法打开视频: {filename}")
            continue

        # 获取总帧数（仅显示用）
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  原视频总帧数: {total_frames}")

        current_frame_idx = 0   # 当前读取到第几帧（从0开始）
        saved_img_count = 1     # 保存的文件名编号（从1开始）

        while True:
            ret, frame = cap.read()
            
            # 视频读完则退出
            if not ret:
                break

            # 逻辑：只有当帧数能被 10 整除时才保存
            # 即保存第 0, 10, 20, 30... 帧
            if current_frame_idx % 2 == 0:
                output_filename = f"{saved_img_count}.png"
                output_path = os.path.join(current_output_folder, output_filename)
                
                cv2.imwrite(output_path, frame)
                saved_img_count += 1
            
            # 无论是否保存，都要增加当前帧的计数
            current_frame_idx += 1

        cap.release()
        print(f"  完成。共提取 {saved_img_count - 1} 张图片。")

    print("\n所有视频处理完毕！")

if __name__ == "__main__":
    process_videos()