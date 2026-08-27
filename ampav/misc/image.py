#!/bin/env python3
from PIL import Image
import numpy as np
import cv2

def is_black_image(img: Image.Image, 
                   luma_threshhold: float=0.1,
                   frame_threshhold: float=0.95) -> bool:
    """Return true if the image is "black"

    Args:
        img (Image.Image): PIL Image to test
        luma_threshhold (float, optional): percentage of luma to count as black. Defaults to 0.1.
        frame_threshhold (float, optional): percentage of pixels in the frame that must be below the luma_threshhold. Defaults to 0.95.

    Returns:
        bool: True if the image is black
    """
    pixel_count = img.width * img.height
    black_count = 0
    luma_val = int(luma_threshhold * 256)
    for p in img.convert('L').get_flattened_data():
        if p <= luma_val:
            black_count +=1
    return black_count / pixel_count >= frame_threshhold


def get_dominant_color(img: Image.Image) -> int | tuple:
    """Return the dominant color in the image.  The return type will vary based
    on the colorspace of the image

    Args:
        img (Image.Image): Image to test

    Returns:
        int | tuple: Returns the color which appeared the most
    """
    bins = {}
    for p in img.get_flattened_data():
        if p not in bins:
            bins[p] = 0
        bins[p] += 1
    return max(bins, key=bins.get)


def is_smpte_colorbars(img: Image.Image):
    # we need the image in HSV since we're looking for specific color ranges
    colors = {'white': (0, 0, 75),
              'red': (0, 100, 75),
              'yellow': (60, 100, 75),
              'green': (120, 100, 75),
              'cyan': (180, 100, 75),
              'blue': (240, 100, 75),
              'magenta': (300, 100, 75)}

    def hue(val):
        return max([0, min([255, int(255 * (((360 + val) % 360) / 360))])])

        
    def scale(val):
        return max([0, min([255, int(255 * (val / 100))])])

    def color2range(name: str, hue_t, sat_t, val_t):
        if name == 'white':
            c = colors[name]
            return[(0, scale(c[1] - sat_t), scale(c[2] - val_t)),
                   (255, scale(c[1] + sat_t), scale(c[2] + val_t))]
        else:
            c = colors[name]
            return [(hue(c[0] - hue_t), scale(c[1] - sat_t), scale(c[2] - val_t)),
                    (hue(c[0] + hue_t), scale(c[1] + sat_t), scale(c[2] + val_t))]

    # we're going to quantize the image to 13 colors
    qimg = img.quantize(220)
    hsv = np.array(qimg.convert('HSV'))
    res = {}
    for k, v in colors.items():
        rng = color2range(k, 10, 35, 25)
        #print(k, v, rng)
        if rng[0][0] > rng[1][0]:
            # we wrapped around the hue angle, so we have to split it across
            # the boundary
            mask1 = cv2.inRange(hsv, (rng[0][0], rng[0][1], rng[0][2]), (255, rng[1][1], rng[1][2]))
            mask2 = cv2.inRange(hsv, (0, rng[0][1], rng[0][2]), (rng[1][0], rng[1][1], rng[1][2]))
            #cv2.imshow("Mask 1", mask1)
            #cv2.imshow("Mask 2", mask2)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, *rng)




 
        #cv2.imshow(k, mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        res[k] = []
        for cnt in contours:
            
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                print(approx.tolist())
            scnt = approx
            
            scnt = cnt.tolist()
            scnt = [x[0] for x in scnt]
            res[k].append(scnt)



            

    return qimg, res

if __name__ == "__main__":
    i = Image.open("/home/bdwheele/work_projects/AMPAV/SMPTE_COLOR_BAR_75.png")
    print(is_smpte_colorbars(i))