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
        return max([0, min([255, int(256 * (((360 + val) % 360) / 360))])])
        
    def scale(val):
        return max([0, min([255, int(256 * (val / 100))])])

    def color2range(name: str, t):
        c = colors[name]
        return [(hue(c[0] - t), scale(c[1] - t), scale(c[2] - t)),
                (hue(c[0] + t), scale(c[1] + t), scale(c[2] + t))]

    
    hsv = np.array(img.convert('HSV'))

    for k, v in colors.items():
        rng = color2range(k, 10)
        print(k, v, rng)
        if rng[0][0] > rng[1][0]:
            # we wrapped around the hue angle
            mask = cv2.inRange(hsv, (hue(0), scale(0), scale(65)), (hue(10), scale(10), scale(85)))

        else:
            mask = cv2.inRange(hsv, *rng)

        cv2.imshow(k, mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            print(cnt)



if __name__ == "__main__":
    i = Image.open("/home/bdwheele/work_projects/AMPAV/SMPTE_COLOR_BAR_75.png")
    print(is_smpte_colorbars(i))