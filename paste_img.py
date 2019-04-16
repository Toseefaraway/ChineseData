from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
import copy
import cv2
import os
import random


class paste_img:
    def __init__(self, img, bg_img_path):
        self.img = img
        self.bg_img_path = bg_img_path
        self.w = img.size[0]
        self.h = img.size[1]
        self.bg = self.create_bg_img()

    def create_bg_img(self):
        bground_list = os.listdir(self.bg_img_path)
        bground_choice = random.choice(bground_list)
        bground = Image.open(self.bg_img_path + bground_choice)
        x, y = random.randint(0, int((bground.size[0] - self.w) / 12 + 2)
                              ), random.randint(0, int((bground.size[1] - self.h) / 6 + 2))
        bground = bground.crop((x, y, x + self.w, y + self.h))
        return bground

    def paste_img(self):
        text_img = self.img
        bg_img = self.bg
        img_bg = bg_img.load()
        img_text = text_img.load()
        for i in range(text_img.size[0]):
            for j in range(text_img.size[1]):
                if img_text[i, j] >= (150, 150, 150):
                    continue
                else:
                    img_bg[i, j] = img_text[i, j]
        bg_img.convert('RGB')
        return bg_img
