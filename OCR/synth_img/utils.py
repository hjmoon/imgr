import math
import numpy as np

def rotate_pos_2d(pos, radians):
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    rotated_x = pos_x * math.cos(radians) + pos_y * math.sin(radians)
    rotated_y = -pos_x * math.sin(radians) + pos_y * math.cos(radians)
    return rotated_x, rotated_y

def translate_pos_2d(pos, x, y):
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    return pos_x+x, pos_y+y

def get_left_roi(pos):
    return np.min(pos[:, 0])

def get_right_roi(pos):
    return np.max(pos[:, 0])

def get_top_roi(pos):
    return np.min(pos[:, 1])

def get_bottom_roi(pos):
    return np.max(pos[:, 1])

def get_size_roi(pos):
    width = get_right_roi(pos) - get_left_roi(pos)
    height = get_bottom_roi(pos) - get_top_roi(pos)
    return width, height

def get_corpus(filename, limit=20):
    f = open(filename, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    ret = []
    for line in lines:
        line = line.rstrip()
        ret.append(line[:limit])
        # if len(line) < limit:
        #     ret.append(line)
        # word_list = line.rstrip().split()
        # for word in word_list:
        #     ret.append(word)
    return ret