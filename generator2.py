from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import os
import cv2
import re
import argparse
import textwrap
"""
定义一个生成图片的类,输入字体,背景图像,字符串

定义一个作用函数类
"""
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


def to_dictionary(text_path='', code='utf-8'):
    with open(text_path, 'rb') as file:
        info_list = [part.decode(code, 'ignore').strip() for part in file.readlines()]
        string = ''.join(info_list)
        cn_ = re.compile(r'[^\u4e00-\u9fa5]')
        digit_ = re.sub("\D", '', string)
        en_ = ''.join(re.findall(r'[A-Za-z]', string))
        zh = ''.join(cn_.split(string)).strip()
        setting_digit = set(digit_)
        setting_en = set(en_)
        setting_cn = set(zh)
        dictionary_digit = {key: value for key, value in enumerate(setting_digit)}
        dictionary_cn = {key: value for key, value in enumerate(setting_cn)}
        dictionary_en = {key: value for key, value in enumerate(setting_en)}
        return dictionary_cn, dictionary_en, dictionary_digit

# 从文字库中随机选择n个字符
def sto_choice_from_info_str(dictionary, quantity_B):
    cn,en,digit = dictionary
    cn_list = list(cn.keys())
    en_list = list(en.keys())
    digit_list = list(digit.keys())
    cn_choice = np.random.choice(cn_list,np.random.randint(8,11))
    en_choice = np.random.choice(en_list,np.random.randint(3,6))
    digit_choice = np.random.choice(digit_list,np.random.randint(1,5))
    word_list = []
    for c in cn_choice:
        if c in cn.keys():
            word_list.append(cn[c])
    for e in en_choice:
        if e in en.keys():
            word_list.append(en[e])
    for d in digit_choice:
        if d in digit.keys():
            word_list.append(digit[d])
    random.shuffle(word_list)
    random_word = ''.join(word_list)
    # cn_len = len(dictionary)
    # choice_list = np.arange(cn_len)
    # str_choice = []
    # np.random.choice()
    # global random_word
    # if quantity_B == False:
    #     quantity = 10
    #     # key_potion = np.random.randint(0, cn_len - quantity)
    #     key_choice = np.random.choice(choice_list,quantity)
    #     for i_key in key_choice:
    #         if i_key in dictionary.keys():
    #             str_choice.append(dictionary[i_key])
    #             random_word = ''.join(str_choice)
    # else:
    #     quantity = np.random.randint(4,11)
    #     key_choice = np.random.choice(choice_list,quantity)
    #
    #     for i_key in key_choice:
    #         if i_key in dictionary.keys():
    #             str_choice.append(dictionary[i_key])
    #             random_word = ''.join(str_choice)
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

# 旋转函数
def rad(x):
    return x * np.pi / 180
def rotate_func(img):
    rotate = np.random.randint(80,100)
    im2 = img.convert('RGBA')
    im2 = im2.rotate(rad(rotate),expand=True)
    fff = Image.new('RGBA',im2.size,(255,)*4)
    img = Image.composite(im2,fff,im2)
    return img


# 噪声函数
def random_noise_func(img):
    img = np.asarray(img)
    img.flags.writeable =True
    for k in range(70):
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
        img= img.resize((int(w*1.3),int(h)))

    elif type == 2:
        img = img.resize((int(w),int(h*1.3)))

    elif type == 3:
        img = img.resize((int(w*1.3),int(h*1.3)))
    else:
        return img
    return img

# 随机选取文字贴合起始的坐标, 根据背景的尺寸和字体的大小选择
def random_x_y(bground_size, font_size):
    width, height = bground_size
    # 为防止文字溢出图片，x，y要预留宽高
    x = random.randint(0, width-font_size*10)
    y = random.randint(0, int((height-2*font_size)/2))
    return x, y

def random_font_size():
    font_size = random.randint(24,27)
    return font_size
def random_font(font_path):
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)
    return font_path + random_font

def creat_img(bg_path,info_str,font_path,quantity_B=True):
    random_word  = sto_choice_from_info_str(info_str,quantity_B)
    font_size = random_font_size()
    raw_image = create_an_image(bg_path, font_size * 10, font_size*3)
    font_name = random_font(font_path)
    font_color = random_word_color()
    draw_x, draw_y = random_x_y(raw_image.size, font_size)
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(raw_image)
    #调整random_word 进行换行
    para = textwrap.wrap(random_word,10)
    pad = 2
    for line in para:
        w,h = draw.textsize(line,font=font)
        draw.text((draw_x, draw_y), line, fill=font_color, font=font)
        draw_y += pad + h


    return raw_image,random_word
# 选取作用函数
def random_choice_in_process_func(img):
    type = [0,1,2,3,4,5,]
    type =np.random.choice(type,size=np.random.randint(0,5))
    img = random_noise_func(img)
    erode_or_dilate = False
    for i_type in type:
        if i_type == 0:
            img = rotate_func(img)
        if i_type == 1:
            img = darken_func(img)
        if i_type == 2:
            img = stretching_func(img)
        if i_type == 3 and erode_or_dilate == False:
            img = add_dilate(img)
            erode_or_dilate=True
        if i_type == 4 and erode_or_dilate == False:
            img = add_erode(img)
        else:
            img = img
    return img

def main(save_path, num, dictionary,quantity_B):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    raw_image,random_word= creat_img(bg_path='./background/',info_str=dictionary,font_path='./font/',quantity_B=quantity_B)
    raw_image = random_choice_in_process_func(raw_image)

    # 保存文本信息和对应图片名称
    raw_image.save(os.path.join(save_path, random_word+'_'+str(num)+'.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', '--i', type=str, default="./data/m_char_std_5995.txt",
                        help='input Chinese words list file path')
    parser.add_argument('--output', '--o', type=str, default="./out/train/",
                        help='output image files directory')
    parser.add_argument('--num', '--n', type=int, default=100,
                        help='number of output files')
    args = parser.parse_args()

    # open file
    file_name = args.input
    output_path = args.output
    total = args.num
    dictionary = to_dictionary(file_name)
    # with open(file_name, 'r', encoding='utf-8') as input_file:
    #     info_list = [part.strip().replace('\t', '') for part in input_file.readlines()]
    #     info_str = ''.join(info_list)


    for num in range(0, total):
        main(output_path, num,dictionary,quantity_B=False)
        if num % 1000 == 0:
            print('[%d/%d]'%(num,total))
