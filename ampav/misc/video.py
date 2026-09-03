#!/bin/env python3

import logging
import time
from ampav.core.logging import LOG_FORMAT
from ampav.core.schema.tool import ToolOutput
from ampav.misc.image import is_smpte_colorbars
from ampav.misc.audio import audio_fft
from ampav.core.media import get_frames_from_video
from ampav.core.render import render_html
from ampav.core.schema.av_metadata import AVMetadata
import argparse
from pathlib import Path
from itertools import groupby
from . import __version__


def detect_colorbars(filename: Path, min_length: float=3, 
                     max_gap: float=1, silence_db: float=-40):
    results = {}
    media_meta = AVMetadata.from_file(filename)
    media_duration = int(media_meta.duration)

    tool_output = ToolOutput(tool_name="misc-detect_colorbars",
                             tool_version=__version__,
                             start_time=time.time())
    tool_output.setup_logging()

    logging.info(f"Detecting tone frequencies, duration: {media_duration}")
    # look for tone 
    for k, v in audio_fft(filename, 0).items():
        results[int(k)] = {'dominant_frequency': v[0].frequency,
                           'dominant_frequency_db': v[0].db}

    logging.info("loading frames")
    for k, v in get_frames_from_video(filename, 0, list(range(media_duration))).items():        
        logging.info(f"Processing frame {k}")
        if k not in results:
            results[k] = {'dominant_frequency': 0,
                          'dominant_frequency_db': -100}
        results[k]['frame'] = v
        res, tests, qimg = is_smpte_colorbars(v.image)
                
        results[k]['id_frame'] = qimg

        results[k]['tests'] = tests
        results[k]['res'] = res

    # now with a set of results we have to do some work to convert them into
    # ranges
    positive_frames = sorted([x for x in results.keys() if 'res' in results[x] and results[x]['res']])
    frame_groups = []
    # ok, groupby is weird -- it works more like a `uniq` command than a 
    # sql `group by`.  BUT, we can still abuse that:  numbers are in consecutive
    # order if their value - index are the same:
    #  enumerate([2,3,5]) => [[0,2], [1,3], [2,5] -> [2-0, 3-1, 5-2] => 2, 2, 3 
    # so the first two are consecutive, but the 3rd isn't.
    for group in groupby(enumerate(positive_frames), key=lambda x: x[1] - x[0]):
        group = list(group[1])        
        start = group[0][1] - 1  # frames & audio happen after the first second
        end = group[-1][1]
        frame_groups.append((start, end))

    # merge the frames that have less than min_gap distance between them.
    merged_groups = []
    if not frame_groups:
        final_res = []
    else:
        cur = frame_groups.pop(0)
        while len(frame_groups):
            this = frame_groups.pop(0)
            if this[0] - cur[1] <= max_gap:
                cur = (cur[0], this[1])
            else:
                merged_groups.append(cur)
                cur = this
        merged_groups.append(cur)

        # filter out groups that are less than the 
        filtered_groups = []
        for group in merged_groups:
            if group[1] - group[0] >= min_length:
                filtered_groups.append(group)

        # build the segments AND determine the audio that's present
        final_res = []
        for start, end in filtered_groups:
            tones = [results[k]['dominant_frequency'] for k in results.keys() if start <= k <= end and results[k]['dominant_frequency_db'] >= silence_db and results[k]['dominant_frequency'] != 0]
            max_freq = max(tones)
            min_freq = min(tones)
            avg_freq = sum(tones) / len(tones)
            final_res.append({'start_time': start,
                                                'end_time': end,
                                                'min_freq': min_freq,
                                                'avg_freq': avg_freq,
                                                'max_freq': max_freq})

    results['colorbar_segments'] = final_res
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
    logging.info("Rendering")
    args.outfile.write_text(render_html(detect_colorbars(args.infile),str(args.infile)))