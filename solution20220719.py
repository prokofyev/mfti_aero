import numpy as np

from PIL import Image, ImageEnhance
import cv2

import glob
import os.path

import json
import re
import math
import tqdm


TEST_PATH = "test_dataset_test/test/"

SCALE = 20
SCALE2 = 10


def process(im):
    im = np.array(im)
    return Image.fromarray(cv2.adaptiveThreshold(im, 255, 
             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 6))   


def process_clouds(im):
    AREA_THRESHOLD = 100
    
    im = ImageEnhance.Brightness(im).enhance(0.7)
    im = ImageEnhance.Contrast(im).enhance(7)
  
    mask = np.ones(np.array(im).shape[:2], dtype="uint8") * 255
    for t in range(140, 255, 1):
        im2 = cv2.inRange(np.array(im), t, t + 1)
        cnts, _ = cv2.findContours(im2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > AREA_THRESHOLD:
                cv2.drawContours(mask, [c], -1, 0, -1)
            
    mask = Image.fromarray(mask)    
    return im, mask


def dist(n, n2, mask=None):
    
    def process(n):
        n = np.array(n, dtype=float)
        if mask is not None:
            n = n[np.array(mask) == 255]
        n = n.flatten()
        return n
    
    n = process(n)
    n2 = process(n2)
    
    return np.power(n - n2, 2).sum()


def has_black_corner(im):
    LEN = im.size[0] // 7
    im = np.array(im)
    return (im[:LEN, :LEN] == 0).all() or \
            (im[:LEN, -LEN:] == 0).all() or \
            (im[-LEN:, :LEN] == 0).all() or \
            (im[-LEN:, -LEN:] == 0).all()


def insert_min(min_dist, dist, k):
    i = 0
    while i < len(min_dist) and min_dist[i][0] <= dist[0]:
        i += 1
        
    if i < k:
        min_dist = min_dist[:i] + [dist] + min_dist[i:k - 1]
        
    return min_dist


def coords(dist):
    size_r = im_s2.rotate(dist[1], expand=True).size[0]
    
    left_top_x, left_top_y = dist[2] - size_r // 2, dist[3] - size_r // 2
    right_top_x, right_top_y = dist[2] + size_02 - size_r // 2, dist[3] - size_r // 2
    left_bottom_x, left_bottom_y = dist[2] - size_r // 2, dist[3] + size_02 - size_r // 2
    right_bottom_x, right_bottom_y = dist[2] + size_02 - size_r // 2, dist[3] + size_02 - size_r // 2
    
    angle = dist[1] * 2 * math.pi / 360
    
    return {
            "left_top": [int((left_top_x * math.cos(angle) - left_top_y * math.sin(angle) + size2 // 2) * SCALE2), 
                        int((left_top_x * math.sin(angle) + left_top_y * math.cos(angle) + size2 // 2) * SCALE2)],
            "right_top": [int((right_top_x * math.cos(angle) - right_top_y * math.sin(angle) + size2 // 2) * SCALE2), 
                        int((right_top_x * math.sin(angle) + right_top_y * math.cos(angle) + size2 // 2) * SCALE2)],
            "left_bottom": [int((left_bottom_x * math.cos(angle) - left_bottom_y * math.sin(angle) + size2 // 2) * SCALE2), 
                        int((left_bottom_x * math.sin(angle) + left_bottom_y * math.cos(angle) + size2 // 2) * SCALE2)],
            "right_bottom": [int((right_bottom_x * math.cos(angle) - right_bottom_y * math.sin(angle) + size2 // 2) * SCALE2), 
                        int((right_bottom_x * math.sin(angle) + right_bottom_y * math.cos(angle) + size2 // 2) * SCALE2)],
            "angle": dist[1],
           }


im = Image.open("train_dataset_train/original.tiff")

size = im.size[0] // SCALE
size2 = im.size[0] // SCALE2

im_s = im.convert('L').resize((size, size))
im_s2 = im.convert('L').resize((size2, size2))

im_s_c = process(im_s)
im_s2_c = process(im_s2)

ANGLE_STEP = 5

im_s_rs = {}
im_s_c_rs = {}
im_s2_c_rs = {}
im_s2_rs = {}

for angle in range(-ANGLE_STEP, 360 + ANGLE_STEP):
    if angle not in im_s_c_rs:
        im_s_rs[angle] = im_s.rotate(angle, expand=True)
        im_s_c_rs[angle] = im_s_c.rotate(angle, expand=True)
        im_s2_rs[angle] = im_s2.rotate(angle, expand=True)
        im_s2_c_rs[angle] = im_s2_c.rotate(angle, expand=True)
        
path = TEST_PATH + "img/"

STEP = 7
MIN_K = 5

W = 0.6
    
for fn in tqdm.tqdm(glob.glob(path + "*.png")):
    out_fn = fn.replace("/img/", "/json/").replace(".png", ".json")
    if os.path.isfile(out_fn):
        continue
    
    im_0 = Image.open(fn)
    im_0 = im_0.convert('L')
    
    if int(re.findall("\d+", fn)[0]) > 800:
        im_0, mask_0 = process_clouds(im_0)
    else:
        mask_0 = None
        
    size_0 = im_0.size[0] // SCALE
    size_02 = im_0.size[0] // SCALE2

    im_0_s = im_0.resize((size_0, size_0))
    im_02_s = im_0.resize((size_02, size_02))  
    if mask_0 is not None:
        mask_0_s = mask_0.resize((size_0, size_0))
        mask_02_s = mask_0.resize((size_02, size_02)) 
    else:
        mask_0_s = None
        mask_02_s = None

    im_0_s_c = process(im_0_s)
    im_02_s_c = process(im_02_s)

    step = STEP
    angle_step = ANGLE_STEP

    min_dist2 = []

    while step > 1:

        min_dist = []

        for angle in tqdm.tqdm(range(0, 360, angle_step), leave=False, desc="Rotating..."):
            im_s_c_r = im_s_c_rs[angle]
            im_s_r = im_s_rs[angle]

            for top in range(0, im_s_c_r.size[1] - size_0, step):
                for left in range(0, im_s_c_r.size[0] - size_0, step):
                    cr = im_s_c_r.crop((left, top, left + size_0, top + size_0))
                    if not has_black_corner(cr):
                        d = dist(cr, im_0_s_c, mask_0_s) + \
                            dist(im_s_r.crop((left, top, left + size_0, top + size_0)), im_0_s, mask_0_s) * W
                        min_dist = insert_min(min_dist, (d, angle, left, top), MIN_K)

        for i in tqdm.tqdm(range(len(min_dist)), leave=False, desc="Adjusting..."):
            min_dist2_ = []

            for angle in range(min_dist[i][1] - ANGLE_STEP, min_dist[i][1] + ANGLE_STEP):
                im_s2_c_r = im_s2_c_rs[angle]
                im_s2_r = im_s2_rs[angle]

                for top in range((min_dist[i][3] - STEP) * SCALE // SCALE2, (min_dist[i][3] + STEP) * SCALE // SCALE2):
                    for left in range((min_dist[i][2] - STEP) * SCALE // SCALE2, (min_dist[i][2] + STEP) * SCALE // SCALE2):
                        d = dist(im_s2_c_r.crop((left, top, left + size_02, top + size_02)), im_02_s_c, mask_02_s) + \
                            dist(im_s2_r.crop((left, top, left + size_02, top + size_02)), im_02_s, mask_02_s) * W
                        min_dist2_ = insert_min(min_dist2_, (d, angle, left, top), 1)

            min_dist2 = insert_min(min_dist2, min_dist2_[0], MIN_K)

        if min_dist2[0][0] / min_dist2[1][0] < 0.9:
            break

        step -= 2
        angle_step -= 1
                        
    with open(out_fn, "w") as f:
        json.dump(coords(min_dist2[0]), f)
