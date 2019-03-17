from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import glob
import numpy as np
import os
import cv2
import argparse
import pickle
from argparse import RawDescriptionHelpFormatter
import fnmatch
import json
import  shutil

import traceback
import copy

'''
1. 从文字库随机选择10个字符/从文字库中随机抽出字符(构建字典：ID=>汉字)
2. 生成图片/生成字体(白底黑体)图像,背景图像,之后将字体图像帖和到背景图像中
    a.生成单个字体图像(包含多个字符)
        对图像进行旋转,模糊,腐蚀处理
        对图像进行等比例缩放
        查找包含字体的最小矩形(去除空余部分,使字体图像最小)
    b.贴合到背景图像的固定位置(背景图像从背景文件中选取):数组比大小,取小值
3. 随机使用函数
'''

# 从文字库中随机选择n个字符

def sto_choice_from_info_str(info_str, quantity=10):
    start = random.randint(0, len(info_str)-11)
    end = start + 10
    random_word = info_str[start:end]
    return random_word

def random_word_color():
    # 目前是黑色,更改这个函数还会改变字的颜色
    font_color_choice = [[54,54,54],[54,54,54],[105,105,105]]
    font_color = random.choice(font_color_choice)
    noise = np.array([random.randint(0,10),random.randint(0,10),random.randint(0,10)])
    font_color = (np.array(font_color) + noise).tolist()
    return tuple(font_color)

# 生成一张图片
def create_an_image(bground_path, width, height):
    bground_list = os.listdir(bground_path)
    bground_choice = random.choice(bground_list)
    bground = Image.open(bground_path+bground_choice)
    x, y = random.randint(0,bground.size[0]-width), random.randint(0, bground.size[1]-height)
    bground = bground.crop((x, y, x+width, y+height))
    return bground


# 模糊函数
def darken_func(image):
    #.SMOOTH
    #.SMOOTH_MORE
    #.GaussianBlur(radius=2 or 1)
    # .MedianFilter(size=3)
    # 随机选取模糊参数
    filter_ = random.choice(
                            [ImageFilter.SMOOTH,
                            ImageFilter.SMOOTH_MORE,
                            ImageFilter.GaussianBlur(radius=1.3)]
                            )
    image = image.filter(filter_)

    return image


# 旋转函数/cv2仿射里有旋转
def rad(x):
    return x * np.pi / 180
def rotate_func(img):
    rotate = np.random.randint(100,300)
    img = img.rotate(rad(rotate),expand=True)
    return img


# 噪声函数
def random_noise_func(img):
    img = np.asarray(img)
    img.flags.writeable =True
    for k in range(100):
        i = int(np.random.random()*img.shape[1])
        j = int(np.random.random()*img.shape[0])
        img[j,i,0] = random.randint(0,127)
        img[j,i,1] = random.randint(0,127)
        img[j,i,2] = random.randint(0,127)
    img = Image.fromarray(img)
    return img
def add_erode(img):
    img = np.asarray(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    erode_img = cv2.erode(img,kernel=kernel)
    img = Image.fromarray(erode_img)
    return img
def add_dilate(img):
    img = np.asarray(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilate_img = cv2.dilate(img, kernel=kernel)
    img = Image.fromarray(dilate_img)
    return img

# 字体拉伸函数

def stretching_func(img):
    type = np.random.choice([0,1,2,3])
    w, h = img.size
    if type == 1 :
        img= img.resize((int(w*1.2),int(h)))

    elif type == 2:
        img = img.resize((int(w),int(h*1.2)))

    elif type == 3:
        img = img.resize((int(w*1.2),int(h*1.2)))
    else:
        return img
    return img


# def stretching_func(img):
#     img = np.asarray(img)
#     w,h,_ = img.shape
#     fov = 42
#     # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
#     z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
#     # 齐次变换矩阵
#     rx = np.array([[1, 0, 0, 0],
#                    [0, np.cos(rad(random.random()*10)), -np.sin(rad(random.random()*10)), 0],
#                    [0, -np.sin(rad(random.random()*10)), np.cos(rad(random.random()*10)), 0, ],
#                    [0, 0, 0, 1]], np.float32)
#
#     ry = np.array([[np.cos(rad(random.random()*10)), 0, np.sin(rad(random.random()*10)), 0],
#                    [0, 1, 0, 0],
#                    [-np.sin(rad(random.random()*10)), 0, np.cos(rad(random.random()*10)), 0, ],
#                    [0, 0, 0, 1]], np.float32)
#
#     rz = np.array([[np.cos(rad(random.random()*10)), np.sin(rad(random.random()*10)), 0, 0],
#                    [-np.sin(rad(random.random()*10)), np.cos(rad(random.random()*10)), 0, 0],
#                    [0, 0, 1, 0],
#                    [0, 0, 0, 1]], np.float32)
#
#     r = rx.dot(ry).dot(rz)
#
#     # 四对点的生成
#     pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
#
#     p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
#     p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
#     p3 = np.array([0, h, 0, 0], np.float32) - pcenter
#     p4 = np.array([w, h, 0, 0], np.float32) - pcenter
#
#     dst1 = r.dot(p1)
#     dst2 = r.dot(p2)
#     dst3 = r.dot(p3)
#     dst4 = r.dot(p4)
#
#     list_dst = [dst1, dst2, dst3, dst4]
#
#     org = np.array([[0, 0],
#                     [w, 0],
#                     [0, h],
#                     [w, h]], np.float32)
#
#     dst = np.zeros((4, 2), np.float32)
#
#     # 投影至成像平面
#     for i in range(4):
#         dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
#         dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
#
#     warpR = cv2.getPerspectiveTransform(org, dst)
#     output = cv2.warpPerspective(img, warpR, (h, w))
#
#     # martrix = cv2.getRotationMatrix2D((cols/2,rows/2),10,1.1)
#     # output = cv2.warpAffine(img,martrix,(cols,rows))
#     img = Image.fromarray(output)
#     return img

# 随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_x_y(bground_size, font_size):
    width, height = bground_size
    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width-font_size*10)
    y = random.randint(0, int((height-font_size)/2))
    return x, y



def random_font_size():
    font_size = random.randint(24,27)
    return font_size

def random_font(font_path):
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)
    return font_path + random_font
def creat_img(bg_path,info_str,font_path):
    random_word = sto_choice_from_info_str(info_str, 10)
    # 生成一张背景图片，已经剪裁好，宽高为32*280
    font_size = random_font_size()
    raw_image = create_an_image(bg_path, font_size * 10 + 10, font_size + 10)

    # 随机选取字体大小

    # 随机选取字体
    font_name = random_font(font_path)
    # 随机选取字体颜色
    font_color = random_word_color()
    # 随机选取文字贴合的坐标 x,y
    draw_x, draw_y = random_x_y(raw_image.size, font_size)

    # 将文本贴到背景图片
    font = ImageFont.truetype(font_name, font_size)

    draw = ImageDraw.Draw(raw_image)
    draw.text((draw_x, draw_y), random_word, fill=font_color, font=font)
    return raw_image,random_word
# 选取作用函数
def random_choice_in_process_func(img):
    type = [0,1,2,3,4,5,]
    type =np.random.choice(type,size=np.random.randint(0,6))
    img = random_noise_func(img)
    erode_or_dilate = False
    for i_type in type:
        if i_type == 0 :
            img = random_noise_func(img)
        if i_type == 1 :
            img = rotate_func(img)
        if i_type == 2 :
            img = darken_func(img)
        if i_type == 3:
            img = stretching_func(img)
        if i_type == 4 and erode_or_dilate == False:
            img = add_dilate(img)
            erode_or_dilate=True
        if i_type ==5 and erode_or_dilate == False:
            img = add_erode(img)
        else:
            img = img
    return img

def main(save_path, num, info_str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    raw_image,random_word= creat_img(bg_path='./background/',info_str=info_str,font_path='./font/')
    # 随机选取10个字符
    # random_word = sto_choice_from_info_str(info_str, 10)
    # # 生成一张背景图片，已经剪裁好，宽高为32*280
    # font_size = random_font_size()
    # raw_image = create_an_image('./background/', font_size*10+10, font_size+10)
    #
    # # 随机选取字体大小
    #
    # # 随机选取字体
    # font_name = random_font('./font/')
    # # 随机选取字体颜色
    # font_color = random_word_color()
    # # 随机选取文字贴合的坐标 x,y
    # draw_x, draw_y = random_x_y(raw_image.size, font_size)
    #
    # # 将文本贴到背景图片
    # font = ImageFont.truetype(font_name, font_size)
    #
    # draw = ImageDraw.Draw(raw_image)
    # draw.text((draw_x, draw_y), random_word, fill=font_color, font=font)

    # raw_image = rotate_func(raw_image)
    # raw_image = stretching_func(raw_image)
    # # raw_image = stretching_func(raw_image)
    # raw_image = add_erode(raw_image)
    # raw_image = add_dilate(raw_image)
    #
    # raw_image = random_noise_func(raw_image)
    raw_image = random_choice_in_process_func(raw_image)

    # 保存文本信息和对应图片名称
    raw_image.save(os.path.join(save_path, random_word+'_'+str(num)+'.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', '--i', type=str, default="./data/demo_input.txt",
                        help='input Chinese words list file path')
    parser.add_argument('--output', '--o', type=str, default="./out/train/",
                        help='output image files directory')
    parser.add_argument('--num', '--n', type=int, default=1000,
                        help='number of output files')
    args = parser.parse_args()

    # open file
    file_name = args.input
    output_path = args.output
    total = args.num
    with open(file_name, 'r', encoding='utf-8') as input_file:
        info_list = [part.strip().replace('\t', '') for part in input_file.readlines()]
        info_str = ''.join(info_list)


    for num in range(0, total):
        main(output_path, num, info_str)
        if num % 1000 == 0:
            print('[%d/%d]'%(num,total))
