#!/bin/env python3

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
        qimg, points = is_smpte_colorbars(v.image)
        tmpimg = ImageDraw.Draw(qimg)
        j = 0
        for c, cnt in points.items():            
            for contour in cnt:
                if len(contour) >= 4:
                    p = Polygon(contour)
                    print(c, p.area, p.centroid)
                    tmpimg.text((p.centroid.x, p.centroid.y), str(j), fill=c)
                j += 1
                p2 = None
                p1 = contour[0]
                i = 1
                while i < len(contour):
                    p2 = contour[i]
                    i += 1
                    tmpimg.line((p1, p2), width=1, fill=c)
                    p1 = p2
                if p2 is not None:
                    tmpimg.line((contour[0], p2), width=1, fill=c)
                        
        results[k]['id_frame'] = qimg

        results[k]['contours'] = {clr: len(pts) for clr, pts in points.items()}

        #results[k]['smpte_points'] = points 

    return results
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)
    parser.add_argument("outfile", type=Path)
    args = parser.parse_args()

    args.outfile.write_text(render_html(detect_colorbars(args.infile),str(args.infile)))