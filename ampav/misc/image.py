#!/bin/env python3
from collections import namedtuple
from typing import Any

from PIL import Image, ImageDraw, ImageChops
import numpy as np
import cv2
from shapely import Point, Polygon
import shapely
import logging

def is_black_image(img: Image.Image, 
                   luma_threshold: float=0.1,
                   frame_threshold: float=0.95) -> bool:
    """Return true if the image is "black"

    Args:
        img (Image.Image): PIL Image to test
        luma_threshold (float, optional): percentage of luma to count as black. Defaults to 0.1.
        frame_threshold (float, optional): percentage of pixels in the frame that must be below the luma_threshold. Defaults to 0.95.

    Returns:
        bool: True if the image is black
    """
    pixel_count = img.width * img.height
    black_count = 0
    luma_val = int(luma_threshold * 256)
    for p in img.convert('L').get_flattened_data():
        if p <= luma_val:
            black_count +=1
    return black_count / pixel_count >= frame_threshold


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


def get_dominant_hue(img: Image.Image) -> int:
    """Get the dominant hue angle for an image

    Args:
        img (Image.Image): An image to test

    Returns:
        int: The dominant color's hue angle
    """
    hsvimage = img.convert('HSV')
    bins = {}
    for p in hsvimage.get_flattened_data(0):
        if p not in bins:
            bins[p] = 0
        bins[p] += 1
    return int(360 * (max(bins, key=bins.get) / 255))


def is_smpte_colorbars(img: Image.Image,
                       hue_tolerance: float=10,
                       sat_tolerance: float=25,
                       val_tolerance: float=25,
                       debug_info: dict | None=None) -> bool:
    """Detect whether or not an image is SMPTE color bars (or similar)

    Args:
        img (Image.Image): Frame image to check
        hue_tolerance (float, optional): Hue variation tolerance degrees. Defaults to 10.
        sat_tolerance (float, optional): Saturation variation tolerance percent. Defaults to 25.
        val_tolerance (float, optional): Value variation tolernace percent. Defaults to 25.
        debug_info (dict, optional): If set to a dict, populate with debugging information
        
    Returns:
        bool: True if the image is color bars
    """

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

    # the image needs to have black side bars removed since sometimes we have
    # images where it's a 4:3 on a 16:9 frame (pillarboxed) or there's additional
    # space vertically (such as during vertical retrace)
    # create a pure black frame
    background = Image.new(img.mode, img.size, (0, 0, 0))
    # get the difference
    difference = ImageChops.difference(img, background)
    # subtract our threshold
    difference = ImageChops.add(difference, difference, 2.0, -50)
    # get the bounding box of the content and if it's valid, crop the
    # frame for future operations.
    bbox = difference.getbbox()
    if bbox:
        img = img.crop(bbox)

    # quantize the image and then convert it to an HSV array.
    qimg = img.quantize(220)
    hsv = np.array(qimg.convert('HSV'))

    # create a drawable where we can put our image content and compute the
    # overall area of the image so we can filter out really small polygons and
    # determine our "ideal" section areas.
    debugimg = ImageDraw.Draw(qimg)
    bar_area = (qimg.width / 7) * qimg.height # one full color bar
    bar_width = int(qimg.width / 7)
    bar_order = ['white', 'yellow', 'cyan', 'green', 'magenta', 'red', 'blue']
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
                    #logging.debug(f"{k} poly {found_polys} has area {p.area} which is less than 1/4 of {castellation_area}")
                    continue    
                #logging.debug(f"{k} poly {found_polys} has area {p.area} and center at {p.centroid}")
                res[k].append(p)
                debugimg.polygon(p.exterior.coords, fill=fill_color)
                debugimg.text((p.centroid.x, p.centroid.y), str(found_polys), fill=text_color)
                found_polys += 1

    # Now that we have the polygons, let's see if these things are actually 
    # something resembling a colorbar image
    tests: dict[str, Any] = {}

    # test properties that apply to all colors. this is an inefficient way to 
    # do it, but it allows me to debug it a whole lot easier.    
    for test_num in range(20):
        test_list = []
        test_dict = {}
        # gather the per-color summary
        for color in bar_order:
            match test_num:
                case 0:
                    # Make sure that all of the colors are present.                    
                    #logging.debug(f"{color} has {len(res[color])} polygons")
                    test_list.append(1 if len(res[color]) > 0 else 0)                    
                case 1:
                    # make sure there is at least 1 polygon centered in the 
                    # upper 2/3 of the frame.
                    t = False
                    for p in res[color]:
                        t |= p.centroid.y <= qimg.height * 0.67
                    
                    #logging.debug(f"{color} has a poly in the top 2/3? {t}")
                    test_list.append(1 if t else 0)
                case 2:
                    # the sum of the areas for each polygon centered within
                    # the x range for that bar.
                    for k, v in  {'1/4': 1/4, '1/3': 1/3, '1/2': 1/2, '2/3': 2/3, '3/4': 3/4,
                                  '1/4-67': 0.67 * 1/4, '1/3-67': 0.67 * 1/3, '1/2-67': 0.67 * 1/2, '2/3-67': 0.67 * 2/3, '3/4-67': 0.67 * 3/4,
                                  '1/4-75': 0.75 * 1/4, '1/3-75': 0.75 * 1/3, '1/2-75': 0.75 * 1/2, '2/3-75': 0.75 * 2/3, '3/4-75': 0.75 * 3/4,}.items():
                        t = 0
                        for p in res[color]:
                            #if p.centroid.y <= qimg.height * 0.67:
                            #    t += p.area
                            if bar_order.index(color) * bar_width <= p.centroid.x <= (bar_order.index(color) + 1) * bar_width:
                                t += p.area
                        #logging.debug(f"{color} has area of {t}, vs {bar_area * v}")
                        if k not in test_dict:
                            test_dict[k] = []
                        test_dict[k].append(1 if t >= bar_area * v else 0)
                    
                case 3:
                    # color bar order should be left to right
                    t = 0
                    c = 0
                    for p in res[color]:
                        if p.centroid.y <= qimg.height * 0.67:
                            t += p.centroid.x
                            c += 1
                    if c:
                        #logging.debug(f"{color} has an average x of {t/c}")
                        test_list.append(t / c)
                    else:
                        test_list.append(-1)

                case 4:
                    # check if we have the full white spot.  We're going to look
                    # for a polygon that's at least 50% of the size of the spot
                    # in the adjustment zone, and the center is fairly close to
                    # where we expect it.
                    if color == "white":
                        v = 0
                        for p in res[color]:
                            if p.centroid.y >= qimg.height * 0.75:
                                #logging.debug(f"Polygon {p} is in the adjustment area")
                                if p.area >= adjustment_area * 0.5:
                                    #logging.debug(f"Polygon {p} is at least 50% of the fullwhite")
                                    adj_width = qimg.width / 5.5
                                    dist = shapely.distance(p.centroid, Point(int(1.5 * adj_width), int(qimg.height * 0.87)))
                                    if dist < adj_width / 4:
                                        #logging.debug(f"Distance: {dist}, {adj_width / 4}")
                                        v = 1.0
                                    
                        test_list.append(v)
                case 5:
                    # check for the appropriate colors in the castellation
                    if color in ('blue', 'magenta', 'cyan', 'white'):
                        for p in res[color]:
                            # check for a polygon within the 8%.
                            if qimg.height * 0.67 <= p.centroid.y <= qimg.height * 0.75:
                                #logging.debug(f"{color} Polygon {p} appears in the castellation band")
                                test_list.append(1.0)
                            else:
                                test_list.append(0)
                case 6:
                    # check for the appropriate order of castellation colors
                    if color in ('blue', 'magenta', 'cyan', 'white'):
                        t = 0
                        c = 0
                        for p in res[color]:
                            # check for a polygon within the 8%.
                            if qimg.height * 0.67 <= p.centroid.y <= qimg.height * 0.75:
                                #logging.debug(f"{color} Polygon {p} appears in the castellation band")
                                t += p.centroid.x
                                c += 1
                        if c:
                            test_list.append(t / c)
                        else:
                            test_list.append(-1)

                case 7:
                    # castellation bar area
                    if color in ('blue', 'magenta', 'cyan', 'white'):
                        for k, v in  {'1/4': 1/4, '1/3': 1/3, '1/2': 1/2, '2/3': 2/3, '3/4': 3/4}.items():
                            t = 0
                            for p in res[color]:
                                if qimg.height * 0.67 <= p.centroid.y <= qimg.height * 0.75:
                                    t += p.area
                            logging.debug(f"castellation {color} has area of {t}, vs {castellation_area * v}")
                            if k not in test_dict:
                                test_dict[k] = []
                            test_dict[k].append(1 if t >= castellation_area * v else 0)


        def get_ascending(x: list):
            c = 0
            last = -1
            for v in x:
                if v > last:
                    c += 1
                    last = v
                elif v == -1:
                    c += 1
                else:
                    break
            logging.debug(f"Get ascending: {x} = {c}")
            return c

        # do something with the summary
        match test_num:
            case 0:
                tests['colors_in_frame']= sum(test_list) / 7
            case 1:
                tests['colors_in_top_2/3'] = sum(test_list) / 7
            case 2:                
                tests['primary_bar_area'] = {}
                for k, v in test_dict.items():                    
                    tests['primary_bar_area'][k] = sum(test_dict[k]) / 7    
            case 3:
                #tests['colors_in_correct_order'] = len(test_list)/7 if test_list == sorted(test_list) else 0
                tests['colors_in_correct_order'] = get_ascending(test_list) / 7
            case 4:
                tests['fullwhite_spot'] = test_list[0]
            case 5:
                tests['castellation_colors'] = sum(test_list) / 4
            case 6:
                # the castellation colors are in the opposite order as the main color bars.
                test_list.reverse()
                tests['castellation_order'] = get_ascending(test_list) / 4
            case 7:
                tests['castellation_bar_area'] = {}
                for k, v in test_dict.items():                    
                    tests['castellation_bar_area'][k] = sum(test_dict[k]) / 4


    # Now that we have all of the test data, let's make a determination. 
    # The base cutoff is:  we have to have 5 of the 7 bars in the top 2/3 and
    # those have to be in order.
    res = False
    if tests['colors_in_top_2/3'] >= 5/7 and int(tests['colors_in_correct_order']) == 1:
        # now we have to make sure that those color bars are a reasonable
        # percentage of the area we're expecting. And that really depends on
        # whether or not we have the fullwhite and castellation.  
        if int(tests['fullwhite_spot']) == 1:
            # with a fullwhite spot the color bars can be 75% height if there's
            # no castellation or "normal sized" at 67% if there is.
            # But how do we determine if there's castellation?  I'm going to 
            # say two of the castellation colors have to be there in the
            # right order. That's possible because they're in the opposite order
            # of the main colorbars so there's no confusion.
            if tests['castellation_colors'] >= 0.5 and int(tests['castellation_order']) == 1:
                # the bars are regular size, so we're going to compare against
                # the 1/4, 1/2, and 3/4 sizes
                if all([tests['primary_bar_area']['1/4-67'] >= 5/7,
                        tests['primary_bar_area']['1/2-67'] >= 4/7,
                        tests['primary_bar_area']['2/3-67'] >= 3/7]):
                    res = True
                else:
                    res = False
            else:
                # If we don't have castellation, then the bars may extend to
                # 75% ... so it's the same as above
                if all([tests['primary_bar_area']['1/4-75'] >= 5/7,
                        tests['primary_bar_area']['1/2-75'] >= 4/7,
                        tests['primary_bar_area']['2/3-75'] >= 3/7]):
                    res = True
                else:
                    res = False
                
        else:
            # we're not going to have castellation without a fullwhite so
            # there's a reasonable assumption that the bars extend from
            # top to bottom.
            #  At least 5 bars must be 1/4 of the area, 4 of them should be at 
            # least 1/2 and 3 of them should be 3/4
            if all([tests['primary_bar_area']['1/4'] >= 5/7,
                    tests['primary_bar_area']['1/2'] >= 4/7,
                    tests['primary_bar_area']['2/3'] >= 3/7]):
                res = True
            else:
                res = False
        
    tests['is_colorbars'] = res

    if debug_info is not None:
        debug_info['qc_frame'] = qimg
        debug_info['tests'] = tests

    return tests['is_colorbars']



if __name__ == "__main__":
    i = Image.open("/home/bdwheele/work_projects/AMPAV/SMPTE_COLOR_BAR_75.png")
    print(is_smpte_colorbars(i))
    i = Image.open("/home/bdwheele/Documents/yellow.png")

    print(get_dominant_hue(i))
