#!/bin/env python3
from collections import namedtuple

from PIL import Image, ImageDraw
import numpy as np
import cv2
from shapely import Polygon
import logging

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


def is_smpte_colorbars(img: Image.Image,
                       hue_tolerance: float=10,
                       sat_tolerance: float=15,
                       val_tolerance: float=25):
    # we need the image in HSV since we're looking for specific color ranges
    HSV = namedtuple("HSV", ['h', 's', 'v'])
    ColorSpec = namedtuple('ColorSpec', ['match', 'fg'])
    colors: dict[str, ColorSpec] = {
        'white': ColorSpec(HSV(0, 0, 75), 'white'),
        'red': ColorSpec(HSV(0, 100, 75), 'red'),
        'yellow': ColorSpec(HSV(60, 100, 75), 'yellow'),
        'green': ColorSpec(HSV(120, 100, 75), 'green'),
        'cyan': ColorSpec(HSV(180, 100, 75), 'cyan'),
        'blue': ColorSpec(HSV(240, 100, 75), 'blue'),
        'magenta': ColorSpec(HSV(300, 100, 75), 'magenta')
    }

    def byte_clamp(val: float) -> int:
        """Clamp a 0-1 float value to 0-255"""
        return max(0, min(255, int(255 * val)))

    def scale_hue(val):      
        # NOTE:  the original data is coming from a PIL HSV image and not an
        # openCV image, so the Hue values are 0-255, rather than the the
        # 0-179 that openCV uses.  The mod is because we can wrap around
        # 0/360 degrees.
        return byte_clamp(((360 + val) % 360) / 360)
    
    def scale(val):
        return byte_clamp(val / 100)
    
    def color2range(name: str, hue_t: float, sat_t: float, val_t: float) -> tuple[HSV, HSV]:
        c = colors[name].match
        if name == 'white':
            return (HSV(0, scale(c.s - sat_t), scale(c.v - val_t)),
                    HSV(255, scale(c.s + sat_t), scale(c.v + val_t)))    
        else:
            return (HSV(scale_hue(c.h - hue_t), scale(c.s - sat_t), scale(c.v - val_t)),
                    HSV(scale_hue(c.h + hue_t), scale(c.s + sat_t), scale(c.v + val_t)))
 
    # quantize the image and then convert it to an HSV array.
    qimg = img.quantize(220)
    hsv = np.array(qimg.convert('HSV'))

    # create a drawable where we can put our image content and compute the
    # overall area of the image so we can filter out really small polygons and
    # determine our "ideal" section areas.
    debugimg = ImageDraw.Draw(qimg)
    frame_area = qimg.width * qimg.height  # the whole frame
    bar_area = (qimg.width / 7) * (qimg.height * 0.67)  # area of one color bar
    castellation_area = (qimg.width / 7) * (qimg.height * 0.08) # castellation bar area
    adjustment_area = (qimg.height * 0.25) * (qimg.width / 5.5) # i, fullwhite, q areas


    # build the masks for each of the colors we're looking for and get the
    # polygons.
    res: dict[str, list[Polygon]] = {}
    
    for k in colors:
        fill_color = f"hsv({colors[k].match.h}, {colors[k].match.s}%, {colors[k].match.v}%)"
        text_color = colors[k].fg
        rng = color2range(k, hue_tolerance, sat_tolerance, val_tolerance)
        if rng[0].h > rng[1].h:
            # we wrapped around the 0 degee hue angle, so we have to make two
            # masks and then or them together.            
            mask1 = cv2.inRange(hsv, (rng[0].h, rng[0].s, rng[0].v), (255, rng[1].s, rng[1].v))
            mask2 = cv2.inRange(hsv, (0, rng[0].s, rng[0].v), (rng[1].h, rng[1].s, rng[1].v))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, *rng)
                
        # detect the contours for this color mask and convert them into polygons
        # the RETR_EXTERNAL is just going to do outer shapes and not count
        # holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res[k] = []
        found_polys = 0
        for cnt in contours:
            # Epsilon is the value the precision of the resulting polygon
            # relative to the total perimeter (arcLength).  The value is a float
            # percentage, the higher the number, the fewer jaggies.  2% is the
            # suggested starting point.
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            # get the approximate polygon vertices.
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # we have to have at least 3 vertices for a polygon...anything else
            # is just a line or point.
            if len(approx) >= 3:                
                # flatten the contours matrix and generate a polygon
                p = Polygon([x[0] for x in approx.tolist()])
                # filter out polygons which are less than 1/4 of a castellation
                # blob...which is being pretty generous
                if p.area < castellation_area * 0.25:
                    logging.debug(f"{k} poly {found_polys} has area {p.area} which is less than 1/4 of {castellation_area}")
                    continue    
                logging.debug(f"{k} poly {found_polys} has area {p.area} and center at {p.centroid}")
                res[k].append(p)
                debugimg.polygon(p.exterior.coords, fill=fill_color)
                debugimg.text((p.centroid.x, p.centroid.y), str(found_polys), fill=text_color)
                found_polys += 1

    return res, qimg

if __name__ == "__main__":
    i = Image.open("/home/bdwheele/work_projects/AMPAV/SMPTE_COLOR_BAR_75.png")
    print(is_smpte_colorbars(i))