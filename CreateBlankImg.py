from PIL import Image, ImageDraw, ImageFont, ImageChops
import numpy as np
import copy
import cv2
import os


class CreateBlankImg:
    def __init__(self, font, word):
        self.font = font
        self.word = word

    def random_font_color(self):

        # color_blue_2 = [98, 120, 141]
        # color_green_1 = [61, 86, 73]
        # color_orange_1 = [183, 149, 139]

        a = [0, 1, 2, ]

        color = np.random.choice(a)

        if color == 0:
            color_value = [0, 206, 209]
        if color == 1:
            color_value = [0, 100, 0]
        if color == 2:
            color_value = [238, 118, 0]

        return tuple(color_value)

    def create_blank_img(self):
        null_img = Image.new('RGB', (280, 40), (255, 255, 255))
        draw = ImageDraw.Draw(null_img)
        color = self.random_font_color()
        font = ImageFont.truetype(self.font, 24)
        draw.text((10, 5), self.word, fill=(0, 0, 0), font=font)
        img = null_img.convert('RGB')
        return img
