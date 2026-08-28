#!/bin/env python3

import logging
from ampav.core.logging import LOG_FORMAT
from ampav.misc.image import is_smpte_colorbars
from ampav.misc.audio import audio_fft
from ampav.core.media import get_frames_from_video
from ampav.core.render import render_html
from PIL import ImageDraw
import argparse
from pathlib import Path
from shapely import Polygon

def detect_colorbars(filename: Path):
    results = {}
    for k, v in audio_fft(filename, 0).items():
        results[k] = {'dominant_frequency': v[0].frequency,
                      'dominant_frequency_db': v[0].db}
    for k, v in get_frames_from_video(filename, 0, list(results.keys())).items():        
        results[k]['frame'] = v
        points, qimg = is_smpte_colorbars(v.image)
                   
        results[k]['id_frame'] = qimg

        results[k]['contours'] = {clr: len(pts) for clr, pts in points.items()}

        #results[k]['smpte_points'] = points 

    return results
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    args.outfile.write_text(render_html(detect_colorbars(args.infile),str(args.infile)))