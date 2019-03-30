import binascii
import numpy as np
from PIL import ImageDraw,Image
KEYS = [0x80,0x40,0x20,0x10,0x08,0x04,0x02,0x01]

def get_dianzhen_word_arr(word,dz_path):
    length = len(word)
    rect_list = []
    for i in word :
        gb18030 = word.encode('gb18030')
        hex_str = binascii.b2a_hex(gb18030)
        result = str(hex_str,encoding='utf-8')
        area = eval('0x'+result[:2])-0xA0
        index = eval('0x'+result[2:]) - 0xA0
        offset = (94*(area-1)+(index-1))*32
        font_rect = None
        with open(dz_path,"rb") as f:
            f.seek(offset)
            font_rect = f.read(32)
        for k in range(len(font_rect)//2):
            for j in range(2):
                for i in range(8):
                    asc = font_rect[k*2+j]
                    flag = asc & KEYS[i]
                    rect_list.append(flag)
    dz_list = rect_list*3
    dz_arr = np.array(dz_list)
    dz_arr = dz_arr.reshape((3,16,16*length))
    # dz_arr = dz_arr.reshape((3, 32, 32 * length))
    dz_arr = np.where(dz_arr>0,0,255)
    dz_arr = dz_arr.transpose((1,2,0))

    return dz_arr

def get_str_dz_arr(str,dz_path):
    str_arr = np.zeros(shape=(16,16,3))
    for i in str:
        word_arr =get_dianzhen_word_arr(i,dz_path)

        str_arr = np.append(str_arr,word_arr,axis=1)
    str_arr = str_arr[:,32:,:]

    return str_arr
