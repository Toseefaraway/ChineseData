from CreateBlankImg import *
from Funcs_on_img import *
from paste_img import *
import os
import argparse
import re
import numpy as np
import random


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


def sto_choice_from_info_str(dictionary):
    cn, en, digit = dictionary
    cn_list = list(cn.keys())
    en_list = list(en.keys())
    digit_list = list(digit.keys())
    cn_choice = np.random.choice(cn_list, np.random.randint(6, 8))
    en_choice = np.random.choice(en_list, np.random.randint(2, 3))
    digit_choice = np.random.choice(digit_list, np.random.randint(1, 2))
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
    return random_word


def get_font(font_path):
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)
    return font_path + random_font


def main(save_path, num, dictionary, font_path, bg_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    word = sto_choice_from_info_str(dictionary)
    font = get_font(font_path)
    img = CreateBlankImg(font, word)
    img = img.create_blank_img()
    img = Funcs_on_img(img)
    img = img.random_fun()
    bg_img = paste_img(img, bg_path)
    bg_img.create_bg_img()
    raw_image = bg_img.paste_img()

    raw_image.save(os.path.join(save_path, word + '_' + str(num) + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', '--i', type=str, default="./data/m_char_std_5995.txt",
                        help='input Chinese words list file path')
    parser.add_argument('--output', '--o', type=str, default="./out/train/kaiti/",
                        help='output image files directory')
    parser.add_argument('--num', '--n', type=int, default=100,
                        help='number of output files')
    parser.add_argument('--font', '--f', type=str, default='./font_copy/', help='get the font files')
    parser.add_argument('--bground', '--b', type=str, default='./background/', help='get the background files')

    args = parser.parse_args()

    # open file
    bg_path = args.bground
    font_path = args.font
    file_name = args.input
    output_path = args.output
    total = args.num
    dictionary = to_dictionary(file_name)
    # with open(file_name, 'r', encoding='utf-8') as input_file:
    #     info_list = [part.strip().replace('\t', '') for part in input_file.readlines()]
    #     info_str = ''.join(info_list)

    for num in range(0, total):
        main(output_path, num, dictionary, font_path, bg_path)
        if num % 1000 == 0:
            print('[%d/%d]' % (num, total))
