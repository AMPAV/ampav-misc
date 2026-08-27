#!/bin/env python3

from collections import namedtuple
import math
import time
import logging
import argparse
from ampav.core.logging import LOG_FORMAT
from ampav.core.media import ChunkedAudio, load_and_resample_audio_file
from pathlib import Path
from ampav.core.schema.audio import AudioEffect, AudioEffectType, AudioEffects
from ampav.core.schema.segments import ConfidenceSegment
from ampav.core.schema.tool import ToolOutput
from ampav.core.utils import dump_data
from . import __version__
import numpy as np
from itertools import batched
from PIL import Image, ImageDraw


def detect_silence(media_file: Path, audio_stream: int=0, 
                   window_millis: int=50, silence_db: float=-40,
                   min_silence_duration: float=0.5) -> ToolOutput:
    """Detect silence in audio

    Args:
        media_file (Path): File to process
        audio_stream (int, optional): Audio stream id. Defaults to 0.
        window_millis (int, optional): detection window size. Defaults to 50.
        silence_db (float, optional): Minimum dB for silence. Defaults to -40.
        min_silence_duration (float, optional): Minimum length in seconds of detected silence. Defaults to 0.5.

    Returns:
        ToolOutput: A tool output of AudioEffects which indicates where there is silence
    """
    # detect silence using RMS via numpy
    tool_output = ToolOutput(tool_name='misc-detect-silence',
                             tool_version=__version__,
                             start_time=time.time(),
                             )
    tool_output.setup_logging()

    _, sample_rate, samples = load_and_resample_audio_file(media_file, audio_stream, channels=1)
    chunk_size = int((sample_rate / 1000) * window_millis)
    segments = []
    in_flight = None
    cur_time = 0
    for window in batched(samples, chunk_size):
        # convert to float32
        logging.debug(f"Window: {window[:50]}")
        window_float = np.array(window, np.float32) 
        # the elusive Root-mean-square
        rms = np.sqrt(np.mean(window_float ** 2))
        # convert to dBFS
        power = 20 * np.log10(rms) if rms else 0
        logging.debug(f"{power} = {rms} / {window_float}")
        if power <= silence_db:
            if in_flight:
                in_flight['end_time'] = cur_time
            else:
                in_flight = {'start_time': cur_time,
                             'end_time': cur_time}
        else:
            if in_flight:
                segments.append(in_flight)
                in_flight = None

        cur_time += 1/sample_rate * len(window)
        
    if in_flight:
        segments.append(in_flight)

    # walk through the segments and if the duration of the segment is less than
    # our "min silence" duration then we just drop it.
    
    out = AudioEffects(media_duration=len(samples) / sample_rate)
    tool_output.output = out
    silence = AudioEffect(type=AudioEffectType.SILENCE,
                          label="rms silence")
    out.effects = [silence]
    for s in segments:
        if s['end_time'] - s['start_time'] >= min_silence_duration:
            silence.instances.append(ConfidenceSegment(start_time=s['start_time'],
                                                       end_time=s['end_time']))
    tool_output.end_time = time.time()

    return tool_output


def generate_waveform_image(media_file: Path, audio_stream: int=0, 
                            height: int=1000, width: int=100, 
                            fg_color='green', bg_color='white') -> Image:
    """Create an image representing the waveform of the audio file

    Args:
        media_file (Path): The file to process
        audio_stream (int, optional): The audio stream ID. Defaults to 0.
        height (int, optional): Result image height. Defaults to 1000.
        width (int, optional): Result image width. Defaults to 100.
        fg_color (str, optional): Color used for the waveform. Defaults to 'green'.
        bg_color (str, optional): Color used for the background. Defaults to 'white'.

    Returns:
        Image: A PIL image representing the waveform
    """

    
    _, _, samples = load_and_resample_audio_file(media_file, audio_stream, channels=1)

    chunk_size = math.floor(len(samples) / width)
    values = []
    for x, window in enumerate(batched(samples, chunk_size)):
        window_float = np.array(window) 
        value = np.mean(window_float) * (height / 2)         
        values.append(value)

    h_range = max(values) - min(values)    
    scale = height / h_range
    img = Image.new(mode="RGB", size=(width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # draw the data
    for x, v in enumerate(values):
        draw.line((x, height/2, x, (v * scale) + (height/2)), fill=fg_color, width=1)
        
    return img


FreqResult = namedtuple('FreqResult', ['frequency', 'db'])


def audio_fft(media_file: Path, audio_stream: int=0,
              silence_db: float=-40) -> dict[float, tuple[FreqResult, list[FreqResult]]]:
    """Perform FFT analysis of the audio

    Args:
        media_file (Path): The file to process
        audio_stream (int, optional): the audio stream id. Defaults to 0.
        silence_db (float, optional): Minimum silence in dB. Defaults to -40.

    Returns:
        dict[float, tuple[FreqResult, list[FreqResult]]]: A map of times to a tuple of the dominant frequency and a list of the other frequencies detected
    """
    results = {}
    # resample the audio to mono 44.1KHz so we can detect frequences up to 22.05KHz
    # per the Nyquist frequency rule.
    SAMPLE_RATE = 44100
    with ChunkedAudio(media_file, audio_stream, SAMPLE_RATE, 1) as ca:
        # When we finish this up we want the bin sizes to be equal to 1Hz
        # so what we can do is set the chunk size = to the sample rate.
        # For a chunk size of 44100 that means that we're sampling the average
        # over a second.
        CHUNK_SIZE = SAMPLE_RATE
        for mtime, chunk in ca.get_fixed_chunks(CHUNK_SIZE, True):
            # the samples are coming back as a float from 0-1, we need to
            # convert them to int16 -- by multiplying by (2^16)/2.
            SCALE = 32768
            samples: np._ArrayT = chunk * SCALE
            # We use the Hann window to reduce the spectral leakage of the 
            # data points near the edges of our hard breaks.
            windowed_data = samples * np.hanning(CHUNK_SIZE)

            # compute the fft and get the frequency buckets
            fft_spectrum = np.fft.rfft(windowed_data)
            frequencies = np.fft.rfftfreq(CHUNK_SIZE, d=1/SAMPLE_RATE)

            # Get the magnitudes and convert to dBFS
            magnitudes = np.abs(fft_spectrum)  # get rid of the imaginary part
            dbs = 20 * np.log10(magnitudes)    # convert to raw dB
            dbFS = dbs - (20 * np.log10(CHUNK_SIZE * SCALE / 4)) # convert to dB full scale

            # Get the peak index
            peak_index = np.argmax(dbFS)
            dominant_freq = frequencies[peak_index]

            # generate the results for this time period
            if dominant_freq > 0 and dbFS[peak_index] > silence_db:
                results[mtime] = (FreqResult(float(dominant_freq), float(dbFS[peak_index])),
                                    [FreqResult(float(frequencies[i]), float(dbFS[i])) 
                                        for i in range(len(frequencies)) if frequencies[i] != 0 and dbFS[i] > silence_db ])

    return results


def cli_fft():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("file", type=Path, help="File to classify") 
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
    
    result = audio_fft(args.file, 0, silence_db=-40)
    for k, v in result.items():
        print(f"{k}s    {v[0].frequency}Hz, {v[0].db:.3f}dBFS")
        for r in v[1]:
            print(f"    {r.frequency}Hz, {r.db:.3f}dBFS")


def cli_misc_detect_silence():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("file", type=Path, help="File to classify")
    parser.add_argument("output", type=Path, help="Output file")    
    parser.add_argument("--format", choices=['yaml', 'json', 'pickle'], default='yaml', help="Output format, default yaml")
    parser.add_argument("--window", type=int, default=50, help="Processing window, in milliseconds")
    parser.add_argument("--min_silence_duration", type=float, default=0.5, help="Minimum duration of silence to report, in seconds")
    parser.add_argument("--silence_db", type=float, default=-40, help="Maximum silence dB, default -40")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
    
    result = detect_silence(args.file, 0, args.window, args.silence_db, args.min_silence_duration)
    logging.info(f"Saving data to {args.output} in {args.format} format")
    dump_data(result, args.format, args.output)


def cli_misc_waveform_image():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("file", type=Path, help="File to classify")
    parser.add_argument("output", type=Path, help="Output file")    
    parser.add_argument("--width", type=int, default=1000, help="Image width")
    parser.add_argument("--height", type=int, default=100, help="Image height")
    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
    
    result = generate_waveform_image(args.file, 0, height=args.height, width=args.width)
    logging.info(f"Saving data to {args.output} in PNG format")
    result.save(args.output)
