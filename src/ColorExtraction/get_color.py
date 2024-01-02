import os
import cv2
import glob
from PIL import Image
import numpy as np

import wcag_contrast_ratio as contrast

import pandas as pd
from skimage import io

import random

def merge_similar_color(pil_img, target_color):
    ### smooth the image

    # convert to cv2 image so that we can use floodfill later
    cv_img = np.array(pil_img) 
    grayimg = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = grayimg.shape[:2]

    # find the position of tatget color
    seeds = []
    new_value = None
    for i in range(h):
        for j in range(w):
            if cv_img[i][j][0] == target_color[0] and cv_img[i][j][1] == target_color[1] and cv_img[i][j][2]== target_color[2]:
                seeds.append((j, i))
                new_value = int(grayimg[i][j])
    if len(seeds) == 0:
        return

    # only floodfill 30 of them
    sampled_index = random.sample(list(range(len(seeds))), min(len(seeds), 30))
    mask = np.zeros((h+2, w+2), np.uint8)
    for i in sampled_index:
        cv2.floodFill(grayimg, seedPoint=seeds[i], newVal=new_value, loDiff=0, upDiff=10, mask=mask)

    # update the flood area with the target color
    mask = mask[1:h+1, 1:w+1]
    changed_points = []
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                if cv_img[i][j][0] != target_color[0] and cv_img[i][j][1] != target_color[1] and cv_img[i][j][2] != target_color[2]:
                    cv_img[i][j][0] = target_color[0]
                    cv_img[i][j][1] = target_color[1]
                    cv_img[i][j][2] = target_color[2]
                    changed_points.append([i,j])

    im_pil = Image.fromarray(cv_img)
    return im_pil, cv_img


def rgb2luminance(c):
    '''
    used to compute the luminance
    '''
    red, green, blue = c
    return (.2126 * red) + (.7152 * green) + (.0722 * blue) # 2023-08-25 (accessibilityscanner way)
    # return (.299 * red) + (.587 * green) + (.114 * blue)

    # https://www.w3.org/TR/WCAG20-TECHS/G17.html#G17-procedure
    ## seems to be different from the formula I currently use. ???
    ## relative luminance v.s. luminance

def get_most_diff_color(bg_color, colors_sorted, flag=False):
    ## get the color with highest contrat with  bg_color
    ##  get dist
    dist = list(map(lambda x:(np.linalg.norm(np.array(x[1])-np.array(bg_color)), x[1]), colors_sorted[:-1]))
    # each item: [(dist, color), (freq, color)]
    dist_freq = list(zip(dist, colors_sorted))
    dist_sort = list(sorted(dist_freq, key=lambda x:x[0][0]))
    # print("dist:", dist_freq[0], len(dist_freq))
    
    # if flag:
    #     dist_sort = [dis for dis in dist_sort if dis[1][0]> 5]
    return dist_sort

def extract_color(img):
    # img: open by PIL
    w,h = img.size


    colors_sorted = list(sorted(img.getcolors(w*h), key=lambda x:x[0]))
    num_unique = len(colors_sorted)

    # print("before", num_unique)

    # for _, col in [*colors_sorted[-2:], *colors_sorted[0:10]]:
    #     tmp = merge_similar_color(img, col)
    #     if tmp is None:
    #         continue
    #     img, cv_img = tmp
    # colors_sorted = list(sorted(img.getcolors(w*h), key=lambda x:x[0]))
    bg_color = colors_sorted[-1][1]
    bg_lum = rgb2luminance(bg_color) # should be background color
    # print("after", len(colors_sorted))

    if len(colors_sorted) == 1:
        return None, None, bg_color, bg_lum, None #, 1, np.array(img)

    dist_sort = get_most_diff_color(bg_color, colors_sorted)
    if len(dist_sort) == 0:
        return None, None, bg_color, bg_lum, None#, 1, np.array(img)

    # for (_, col),(_,_) in dist_sort[:5]:
    #     tmp = merge_similar_color(img, col)
    #     if tmp is None:
    #         continue
    #     img, cv_img = tmp
    # colors = img.getcolors(w*h)
    # colors_sorted = list(sorted(colors, key=lambda x:x[0]))
    # dist_sort = get_most_diff_color(bg_color, colors_sorted)
    # fg_color = dist_sort[-1][0][1]
    # img, cv_img = merge_similar_color(img, fg_color)

    # colors_sorted = list(sorted(colors, key=lambda x:x[0]))
    # dist_sort = get_most_diff_color(bg_color, colors_sorted, True)

    # if len(dist_sort) == 0:
    #     return None, None, bg_color, bg_lum, None, 1, np.array(img)
    fg_color = dist_sort[-1][0][1]

    # contrast 
    # 2023-8-25 exactly the way accessibility scanner do to calculate the contrast
    con = contrast.rgb([b/255 for b in bg_color], [f/255 for f in fg_color])

    # luminance
    fg_lum = rgb2luminance(fg_color) # should be text color

    return fg_color, fg_lum, bg_color, bg_lum, con#, num_unique, cv_img


def extract(PIL_img):
    # img = io.imread(jpg_filename)
    # PIL_img = Image.fromarray(img)
    # img = np.array(PIL_img)

    fg_color, fg_lum, bg_color, bg_lum, con = extract_color(PIL_img)

    return fg_color, fg_lum, bg_color, bg_lum, con
        


if __name__ == '__main__':
    path = "/media/cheer/UI/Project/DarkPattern/code/individual_modules/ColorExtracter/examples/text/29351_0.jpg"
    import time
    start_time = time.time()
    PIL_img = Image.open(path)
    print(extract_color(PIL_img))
    print("Using", time.time() - start_time)
