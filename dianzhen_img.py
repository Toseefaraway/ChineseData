from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageChops
import random
import numpy as np
import os
import cv2
import re
import argparse
import dianzhen


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
    # en_list = list(en.keys())
    # digit_list = list(digit.keys())
    cn_choice = np.random.choice(cn_list,np.random.randint(10,11))
    # en_choice = np.random.choice(en_list,np.random.randint(2,3))
    # digit_choice = np.random.choice(digit_list,np.random.randint(1,3))
    word_list = []
    for c in cn_choice:
        if c in cn.keys():
            word_list.append(cn[c])
    # for e in en_choice:
    #     if e in en.keys():
    #         word_list.append(en[e])
    # for d in digit_choice:
    #     if d in digit.keys():
    #         word_list.append(digit[d])
    random.shuffle(word_list)
    random_word = ''.join(word_list)
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
def rotate_img(img):
    im2 = img.convert("RGBA")
    rot = im2.rotate(np.random.randint(2,4),expand=True)
    fff = Image.new('RGBA',rot.size,(255,)*4)
    img = Image.composite(rot,fff,rot)
    img.convert('RGB')
    return img
# def rotate_func(img,bg_fill_color):
#     rotate = np.random.randint(80,100)
#     # im2 = img.convert('RGBA')
#     img = img.rotate(rad(rotate),expand=True)
#     img = img.convert('RGB')
#     im = img.load()
#     for i in range(img.size[0]):
#         for j in range(img.size[1]):
#             if im[i,j] ==(0,0,0):
#                 im[i,j] = bg_fill_color
#     return img
    # fff = Image.new('RGB',im2.size,bg_fill_color)
    # fff = fff.convert('RGBA')
    # # img = Image.composite(im2,fff,im2)
    # img = ImageChops.multiply(im2,fff)
    # img.convert('RGB')
    # return img


# 噪声函数
def creat_null_img(word_dz):
    img = Image.new('RGB',(word_dz.shape[1],word_dz.shape[0]),(255,255,255))
    img_arr = np.asarray(img)
    img_arr.flags.writeable = True
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            img_arr[i,j] = word_dz[i,j]
    img = Image.fromarray(img_arr)
    # im = img.load()
    # for i in range(img.size[0]):
    #     for j in range(img.size[1]):
    #         if word_dz[i,j,:] >= 150:
    #             print(word_dz[i,j])
    #             continue
    #         else:
    #             im[i,j] = word_dz[i,j]
    return img
def get_bg_wh(bground):
    width = bground.size[0]
    height = bground.size[1]
    return width,height

def create_img(bground,text_img):
    bg_img = bground
    word_img = text_img.convert('RGB')
    im_bg = bg_img.load()
    im_word = word_img.load()
    for i in range(text_img.size[0]):
        for j in range(text_img.size[1]):
            if im_word[i,j] >= (150,150,150):
                continue
            else:
                im_bg[i,j] = im_word[i,j]
    bg_img.convert('RGB')
    return bg_img

def random_noise_func(img):
    img = img.convert('RGB')
    img = np.asarray(img)
    print(img.shape)
    img.flags.writeable =True
    rows, cols, dims = img.shape
    for i in range(100):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 40
    img = Image.fromarray(img)
    return img
def add_erode(img):
    img = np.asarray(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
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
    y = random.randint(0, int((height-font_size)/2))
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
    # font_name = random_font(font_path)
    word_dz = dianzhen.get_str_dz_arr(random_word,'./HZK/HZK16')
    # font_size = random_font_size()
    word_img =creat_null_img(word_dz)
    #随机选择是否旋转
    ro_or_not = np.random.choice([False,True])
    if ro_or_not ==True:
        word_img = rotate_img(word_img)

    width,height = get_bg_wh(word_img)
    bg_image = create_an_image(bg_path, width,height)
    raw_image = create_img(bg_image,word_img)

    return raw_image,random_word,
# 选取作用函数
def random_choice_in_process_func(img):
    type = [0,1,2]
    type =np.random.choice(type,size=np.random.randint(0,3))
    img = random_noise_func(img)
    # erode_or_dilate=False
    for i_type in type:
        if i_type == 0:
            img = img
        if i_type == 1:
            img = stretching_func(img)
        if i_type == 2:
            img = add_erode(img)
            # erode_or_dilate = True
        # if i_type == 3 and erode_or_dilate==False:
        #     img = add_dilate(img)
    img = darken_func(img)
    return img

def main(save_path, num, dictionary,quantity_B):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    raw_image,random_word= creat_img(bg_path='./background/',info_str=dictionary,
                                     font_path='./font/',quantity_B=quantity_B)
    # raw_image = random_choice_in_process_func(raw_image)

    # 保存文本信息和对应图片名称
    raw_image.save(os.path.join(save_path, random_word+'_'+str(num)+'.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', '--i', type=str, default="./data/ChineseDataGB.txt",
                        help='input Chinese words list file path')
    parser.add_argument('--output', '--o', type=str, default="./out/train/HZK",
                        help='output image files directory')
    parser.add_argument('--num', '--n', type=int, default=100,
                        help='number of output files')
    args = parser.parse_args()

    # open file
    file_name = args.input
    output_path = args.output
    total = args.num
    dictionary = to_dictionary(file_name)


    for num in range(0, total):
        main(output_path, num,dictionary,quantity_B=False)
        if num % 1000 == 0:
            print('[%d/%d]'%(num,total))
