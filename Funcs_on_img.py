from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageChops, ImageFilter
import numpy as np
import copy
import cv2
import os
import skimage
from skimage import util


class Funcs_on_img:
    def __init__(self, img):
        self.img = img

    def random_fun(self):
        funcs_list = [1, 2, 3, 4, 5, 6, 6]
        funcs = np.random.choice(funcs_list, size=np.random.randint(1, 4))
        img = self.img
        for func in funcs:
            if func == 1:
                img = self.streching_func(img)
            if func == 2:
                img = self.rotate_img(img)
            if func == 3:
                img == self.darken_func(img)
            if func == 4:
                img = self.draw_line(img)
            if func == 5:
                img = self.normalize(img)
            if func == 6:
                img = self.ImageEnhance(img)
        return img

    def random_noise_func(self, img):
        img = np.asarray(img)
        img = copy.deepcopy(img)
        img.flags.writeable = True
        SNR = 0.1
        noise_num = int(SNR * img.shape[1] * img.shape[0])
        for k in range(0, noise_num):
            xi = int(np.random.uniform(0, img.shape[0]))
            xj = int(np.random.uniform(0, img.shape[1]))
            if np.random.random() < 0.5:
                img[xi, xj] = 0
            else:
                img[xi, xj] = 255
        # rows, cols, dims = img.shape
        # for i in range(30):
        #     x = np.random.randint(0, rows)
        #     y = np.random.randint(0, cols)
        #     img[x, y, :] = 100
        # img = Image.fromarray(img)
        img = Image.fromarray(np.uint8(img))
        return img

    def streching_func(self, img):
        type = np.random.choice([0, 1, 2, 3])
        img = self.img
        w, h = img.size
        if type == 1:
            img = img.resize((int(w * 1.2), int(h)))
        elif type == 2:
            img = img.resize((int(w), int(h * 1.2)))
        elif type == 3:
            img = img.resize((int(w * 1.2), int(h * 1.2)))
        else:
            img = img
        return img

    def draw_line(self, img):
        w = img.size[0]
        h = img.size[1]
        draw = ImageDraw.Draw(img)
        x = np.random.randint(0, 2)
        if x == 0:
            p = np.random.randint(0, h)
            draw.line(((0, p), (w, p)), fill=(0, 0, 0))
        else:
            p = np.random.randint(0, w)
            draw.line(((p, 0), (p, h)), fill=(0, 0, 0))
        return img

    def darken_func(self, img):
        filter_ = np.random.choice(
            [ImageFilter.SMOOTH,
             ImageFilter.SMOOTH_MORE,
             ImageFilter.GaussianBlur(radius=1.3)]
        )
        img = img.filter(filter_)
        return img

    def rotate_img(self, img):
        # img = self.img
        im2 = img.convert('RGBA')
        rot = im2.rotate(np.random.randint(1, 2), expand=True)
        fff = Image.new('RGBA', rot.size, (255,) * 4)
        img = Image.composite(rot, fff, rot)
        img = img.convert('RGB')
        return img

    def normalize(self, img):
        img = np.asarray(img)
        out = np.zeros(img.shape, np.uint8,)
        cv2.normalize(img, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        img = Image.fromarray(out)
        return img

    def ImageEnhance(self, img):
        enhance_way = [ImageEnhance.Brightness(img), ImageEnhance.Color(
            img), ImageEnhance.Contrast(img), ImageEnhance.Sharpness(img)]
        way = np.random.choice(enhance_way)
        n = 1.5
        img = way.enhance(n)
        return img
