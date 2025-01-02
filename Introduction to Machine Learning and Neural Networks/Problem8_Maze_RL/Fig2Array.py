from PIL import Image
import numpy as np

def remove_black_border(img):
    """移除图片的黑色边框"""
    # 转换为灰度图像数组
    img_array = np.array(img)
    # 找到非黑色像素的行和列索引
    non_black_rows = np.where(img_array.min(axis=1) > 0)[0]
    non_black_cols = np.where(img_array.min(axis=0) > 0)[0]

    # 如果全是黑色，直接返回原图
    if non_black_rows.size == 0 or non_black_cols.size == 0:
        return img

    # 找到有效的区域范围
    top, bottom = non_black_rows[0], non_black_rows[-1]
    left, right = non_black_cols[0], non_black_cols[-1]

    # 裁剪图像
    return img.crop((left, top, right, bottom))

def maze_image_to_array(image_path, block_size=8):
    # 打开图片并转换为灰度图
    img = Image.open(image_path).convert('L')
    
    # 移除黑色边框
    img = remove_black_border(img)
    
    # 将图片大小调整为适合的区块
    img = img.resize((img.width // block_size, img.height // block_size), Image.NEAREST)

    # 将图像转换为数组
    img_array = np.array(img)
    
    # 将像素值转换为0和1
    maze_array = np.where(img_array < 128, 0, 1)  # 黑色为0，白色为1

    return maze_array.tolist()  # 转换为列表形式


maze_array = maze_image_to_array('maze.jpg', block_size=8)
for row in maze_array:
    print(row)

def save_maze_to_txt(maze_array, output_path):
    """将迷宫数组保存到TXT文件"""
    with open(output_path, 'w') as f:
        for row in maze_array:
            f.write(' '.join(map(str, row)) + '\n')

maze_array = maze_image_to_array('maze.jpg', block_size=8)
save_maze_to_txt(maze_array, 'mazeArray.txt')

