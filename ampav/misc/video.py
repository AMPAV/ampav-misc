#!/bin/env python3

import logging
import time
from ampav.core.logging import LOG_FORMAT
from ampav.core.schema.segments import ConfidenceSegment
from ampav.core.schema.tool import ToolOutput
from ampav.core.schema.video import VideoPattern, VideoPatternType, VideoPatterns
from ampav.core.utils import dump_data, duration2hhmmss, size2human
from ampav.misc.image import is_smpte_colorbars
from ampav.misc.audio import audio_fft
from ampav.core.media import get_frames_from_video
from ampav.core.render import render_html
from ampav.core.schema.av_metadata import AVMetadata
import argparse
from pathlib import Path
from itertools import groupby
from PIL import Image, ImageDraw, ImageFont
import math
from . import __version__


def detect_colorbars(filename: Path, min_length: float=3, 
                     max_gap: float=2, silence_db: float=-40,
                     stream_debug_info: dict | None=None):
    results = {}
    media_meta = AVMetadata.from_file(filename)
    media_duration = int(media_meta.duration)

    tool_output = ToolOutput(tool_name="misc-detect_colorbars",
                             tool_version=__version__,
                             start_time=time.time(),
                             parameters={'min_length': min_length,
                                         'max_gap': max_gap,
                                         'silence_db': silence_db})
    tool_output.setup_logging()

    logging.info(f"Detecting tone frequencies, duration: {media_duration}")    
    for k, v in get_frames_from_video(filename, 0, list(range(media_duration))):        
        logging.debug(f"Processing frame {k}")
        if k not in results:
            results[k] = {'dominant_frequency': 0,
                          'dominant_frequency_db': -100}
        results[k]['frame'] = v        
        debug_info = {} if stream_debug_info is not None else None
        res = is_smpte_colorbars(v.image, debug_info=debug_info)                        
        if debug_info is not None:
            stream_debug_info[k] = debug_info
        results[k]['res'] = res

    # look for any tones
    for k, v in audio_fft(filename, 0).items():
        if int(k) not in results:
            results[int(k)] = {}

        results[int(k)].update({'dominant_frequency': v[0].frequency,
                                'dominant_frequency_db': v[0].db})
        
    # now with a set of results we have to do some work to convert them into
    # ranges
    positive_frames = sorted([x for x in results.keys() if results[x].get('res', False)])

    # ok, groupby is weird -- it works more like a `uniq` command than a 
    # sql `group by`.  BUT, we can still abuse that:  numbers are in consecutive
    # order if their value - index are the same:
    #  enumerate([2,3,5]) => [[0,2], [1,3], [2,5] -> [2-0, 3-1, 5-2] => 2, 2, 3 
    # so the first two are consecutive, but the 3rd isn't.
    frame_groups = []
    for group in groupby(enumerate(positive_frames), key=lambda x: x[1] - x[0]):
        group = list(group[1])        
        start = group[0][1] - 1  # frames & audio happen after the first second
        end = group[-1][1]
        frame_groups.append((start, end))

    vps = VideoPatterns(media_duration=media_duration)
    vp = VideoPattern(type=VideoPatternType.COLORBARS,
                        label='bars-and-tone')                

    # merge the frames that have less than min_gap distance between them.
    merged_groups = []
    if frame_groups:
        cur = frame_groups.pop(0)
        while len(frame_groups):
            this = frame_groups.pop(0)
            if this[0] - cur[1] <= max_gap:
                cur = (cur[0], this[1])
            else:
                merged_groups.append(cur)
                cur = this
        merged_groups.append(cur)

        # filter out groups that are less than the minimum length
        filtered_groups = []
        for group in merged_groups:
            if group[1] - group[0] >= min_length:
                filtered_groups.append(group)

        # build the segments AND determine the audio that's present
        for start, end in filtered_groups:
            tones = [results[k]['dominant_frequency'] for k in results.keys() if start <= k <= end and results[k]['dominant_frequency_db'] >= silence_db and results[k]['dominant_frequency'] != 0]
            vp.instances.append(ConfidenceSegment(start_time=start,
                                                  end_time=end,
                                                  tool_private={'min_freq': min(tones),
                                                                'avg_freq': (sum(tones) / len(tones)) if len(tones) else 0,
                                                                'max_freq': max(tones)}))
        vps.patterns.append(vp)
        tool_output.output = vps
        tool_output.end_time = time.time()
    return tool_output


def generate_contact_sheet(filename: Path, stream: int, frame_list: list[float],
                           frame_size: tuple[int, int],
                           page_layout: tuple[int | None, int | None],
                           title: str | None=None, timestamps: bool=True,
                           bgcolor: str | tuple[int, int, int]='white',
                           fgcolor: str | tuple[int, int, int]='black',
                           title_font_size: int=20, timestamp_font_size: int=10
                           ) -> Image.Image:

    if title is None:
        title = filename.name
    title_font = ImageFont.load_default(title_font_size)
    timestamp_font = ImageFont.load_default(timestamp_font_size)

    # compute the dimensions for the page layout.
    if page_layout == (None, None):
        raise Exception("At least one dimension of the page layout must be specified")
    if page_layout[0] is None:
        # very wide.
        frames_per_row = math.ceil(len(frame_list) / page_layout[1])
        frames_per_col = page_layout[1]
    elif page_layout[1] is None:
        # very tall
        frames_per_row = page_layout[0]
        frames_per_col = math.ceil(len(frame_list) / page_layout[0])
    else:
        # fixed size
        if page_layout[0] * page_layout[1] < len(frame_list):
            raise Exception(f"Page layout doesn't have enough spots for the frames.  Have {page_layout[0] * page_layout[1]}, need {len(frame_list)}")
        frames_per_row, frames_per_col = page_layout

    # load the frames into a 2d array
    frames: list[list[Image.Image]] = [[]]
    max_frame_width, max_frame_height = 0, 0
    av_meta =AVMetadata.from_file(filename)    
    for vid_time, frame in get_frames_from_video(filename, stream, [x for x in frame_list if x <= av_meta.duration]):
        # resize the frame within the specified parameters
        frame.image.thumbnail(frame_size)
        if timestamps:
            alpha_image = frame.image.convert('RGBA')
            overlay_base = Image.new('RGBA', (alpha_image.width, alpha_image.height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay_base)
            stamp = duration2hhmmss(vid_time)
            bbox = timestamp_font.getbbox(stamp)
            # draw the alpha box 
            ts_width = bbox[2] - bbox[0]
            ts_height = bbox[3] + 2
            ts_x = overlay_base.width / 2 - ts_width / 2
            ts_y = overlay_base.height - ts_height
            overlay_draw.rectangle((ts_x, ts_y, ts_x + ts_width, ts_y + ts_height), fill=(255, 255, 255, 95))
            overlay_draw.text((ts_x, ts_y), stamp, (0, 0, 0, 255), font=timestamp_font)
            composite = Image.alpha_composite(alpha_image, overlay_base)
            frame.image = composite.convert('RGB')
        max_frame_height = max(frame.image.height, max_frame_height)
        max_frame_width = max(frame.image.width, max_frame_width)

        frames[-1].append(frame.image)
        if len(frames[-1]) == frames_per_row:
            frames.append([])
        
    # generate the title image
    title_bb = title_font.getbbox(title)
    duration_bb = timestamp_font.getbbox(duration_text := f"Duration: {duration2hhmmss(av_meta.duration)}")
    filesize_bb = timestamp_font.getbbox(filesize_text := f"File size: {size2human(av_meta.size)} ({av_meta.size})")
    y_off = title_bb[3] + duration_bb[3] + filesize_bb[3]


    page_image = Image.new('RGB', (frames_per_row * max_frame_width, frames_per_col * max_frame_height + y_off),
                           color=bgcolor)        


    for row_id, row in enumerate(frames):
        for col_id, col in enumerate(row):
            page_image.paste(col, (col_id * max_frame_width, row_id * max_frame_height + y_off))

    page_image_draw = ImageDraw.Draw(page_image)
    page_image_draw.text((0, 0), title, fill=fgcolor, font=title_font)
    page_image_draw.text((0, title_bb[3]), duration_text, fill=fgcolor, font=timestamp_font)
    page_image_draw.text((0, title_bb[3] + duration_bb[3]), filesize_text, fill=fgcolor, font=timestamp_font)

    return page_image

    

def cli_detect_colorbars():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("outfile", type=Path, help="Tool output file")    
    parser.add_argument("--debug", action="store_true", help="Turn on debug logging")
    parser.add_argument("--debugging_file", type=Path, help="Output filename for debugging file")
    parser.add_argument("--min_length", type=float, default=3, help="Minimum colorbar segment length (3 second default)")
    parser.add_argument("--max_gap", type=float, default=2, help="Maximum gap between two color bars for merging (2 second default)")
    parser.add_argument("--silence_db", type=float, default=-40, help="dBFS level for silence (default -40)")
    parser.add_argument("--format", choices=['yaml', 'json', 'pickle'], default='yaml', help="Tool output format")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    debug_info = {} if args.debugging_file else None

    result = detect_colorbars(args.video, min_length=args.min_length, max_gap=args.max_gap, silence_db=args.silence_db, stream_debug_info=debug_info)

    logging.info(f"Saving data to {args.outfile} in {args.format} format")
    dump_data(result, args.format, args.outfile)
        
    if debug_info is not None:
        logging.info(f"Saving debug data to {args.debugging_file} in HTML format")
        args.debugging_file.write_text(render_html(debug_info, str(args.video)))   


def cli_generate_contact_sheet():
    def dim_parser(spec: str, needs_both: bool=False)->tuple:
        if ',' not in spec:            
            raise Exception("You must include a comma with a dimension")
        x, y = spec.split(',')
        res = (int(x) if x != '' else None,
               int(y) if y != '' else None)
        if needs_both and res == (None, None):
            raise Exception("Both X and Y are required")
        return res

    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("outfile", type=Path, help="Contact sheet output file")    
    parser.add_argument("--debug", action="store_true", help="Turn on debug logging")
    parser.add_argument("--format", choices=['png', 'jpeg'], default='png', help="Image output format")
    parser.add_argument("--title", type=str, default=None, help="contact sheet title")
    parser.add_argument("--timestamps", action="store_true", help="Enable timestamp generation")
    parser.add_argument("--bgcolor", default='white', help="Background color (default: white)")
    parser.add_argument("--fgcolor", default="black", help="Foreground color (default: black)")
    parser.add_argument("--title_font_size", default=20, type=int, help="Title Font size")
    parser.add_argument("--timestamp_font_size", default=10, type=int, help="Timestamp Font size")
    parser.add_argument('--page_layout', type=str, default='5,', help="Page dimension as one of: 'x,y', 'x,', or ',y'")
    parser.add_argument('--frame_bounding', type=str, default='200,200', help="Dimension for each frame image")
    subp = parser.add_subparsers(dest="mode", required=True)
    cmd = subp.add_parser("count")
    cmd.add_argument("frame_count", type=int, help="Number of frame images in the contact sheet")
    cmd = subp.add_parser("interval")
    cmd.add_argument("frame_interval", type=float, help="Frame image interval")
    cmd = subp.add_parser("list")
    cmd.add_argument("frame_list", nargs='+', type=float, help="Time offsets for frames")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    try:
        frame_size = dim_parser(args.frame_bounding, True)
    except Exception as e:
        logging.error(f"Error parsing frame_size: {e}")
        exit(1)
    try:
        page_layout = dim_parser(args.page_layout)
    except Exception as e:
        logging.error(f"Error parsing page_layout: {e}")
        exit(1)

    av_meta = AVMetadata.from_file(args.video)
    frame_list = []
    match args.mode:
        case "count" | "interval":            
            if args.mode == 'count':
                interval = av_meta.duration / args.frame_count
            else:
                interval = args.frame_interval            
            off = 0.0
            while off < av_meta.duration:
                frame_list.append(off)
                off += interval
        case "list":
            for ftime in args.frame_list:
                if ftime > av_meta.duration:
                    logging.warning(f"Skipping frame at {ftime} since it is beyond the end of the video {av_meta.duration}")
                    continue
                frame_list.append(ftime)            
            frame_list = sorted(frame_list)

    img: Image.Image = generate_contact_sheet(args.video, 0, frame_list,
                                             frame_size=frame_size,
                                             page_layout=page_layout,
                                             title=args.title, timestamps=args.timestamps,
                                             bgcolor=args.bgcolor, fgcolor=args.fgcolor, 
                                             title_font_size=args.title_font_size,
                                             timestamp_font_size=args.timestamp_font_size)

    img.save(args.outfile, format=args.format)



if __name__ == "__main__":
    #cli_detect_colorbars()
    cli_generate_contact_sheet()