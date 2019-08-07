import numpy as np
import cv2
import os
import glob

def resize(img, size=512, strict=False):
    short = min(img.shape[:2])
    scale = size/short
    if not strict:
        img = cv2.resize(img, (round(
            img.shape[1]*scale), round(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)
    else:
        img = cv2.resize(img, (size,size), interpolation=cv2.INTER_NEAREST)
    return img


def crop(img, size=512):
    try:
        y, x = random.randint(
            0, img.shape[0]-size), random.randint(0, img.shape[1]-size)
    except Exception as e:
        y, x = 0, 0
    return img[y:y+size, x:x+size, :]


def load_image(filename, size=None, use_crop=False):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size:
        img = resize(img, size=size)
    if use_crop:
        img = crop(img, size)
    return img

def get_latest_ckpt(path):
    try:
        list_of_files = glob.glob(os.path.join(path,'*')) 
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except ValueError:
        return None

def save_params(state, params):
    state['model_params'] = params
    return state

def load_params(state):
    params = state['model_params']
    del state['model_params']
    return state, params