import argparse
from datetime import time
import os
import subprocess
import sys

import ffmpeg
from openai import OpenAI
from pysubparser import parser
from pysubparser.classes import subtitle
from pysubparser.cleaners import formatting
from pysubparser.writers import srt

TIMEOUT_RATIO = 2.0  # give the translation call twice as long as the audio length
MAX_TIMEOUT = 30.0 # but don't let it go over 30 seconds

def main():
    parser = argparse.ArgumentParser(description="Transcribe and translate meeting recordings. Requires ffmpeg to be installed. Requires an OpenAI key in $OPENAI_API_KEY")
    
    parser.add_argument("input_file", help="Input file to transcribe", type=str)
    parser.add_argument("-c", "--context", help="Freetext context to help the transcription, useful for teaching phrases/acronyms", type=str, default="")

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print("Input file does not exist")
        exit(1)

    # extract subtitle file
    base = os.path.basename(args.input_file)
    srt_file = f'/tmp/{base}.srt'
    print('\nExtracting subtitles', file=sys.stderr)
    (st, st_err) = (
        ffmpeg
        .input(args.input_file)
        .output(srt_file)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    print('\tDone!', file=sys.stderr)

    # process subtitle file
    proc_subs = process_subtitles(srt_file)

    print('\nTranslating audio', file=sys.stderr)
    oai_client = OpenAI()
    translated_subs = []
    for sub in proc_subs:
        translated_subs.append(get_translation(oai_client, sub, args.input_file, args.context))        

    print('\tDone!', file=sys.stderr)

    out_subs = f'{args.input_file}.srt'
    print(f'\nWriting translated subtitle file to {out_subs}', file=sys.stderr)
    srt.write(subtitles=translated_subs, path=out_subs)
    print('\tDone!', file=sys.stderr)

    print(f'\nAdding translated subtitles to video at {args.input_file}.translated.mp4', file=sys.stderr)
    add_translations(args.input_file, out_subs)
    print('\tDone!', file=sys.stderr)

def process_subtitles(srt_file: str) -> list:
    print('\nProcessing subtitles', file=sys.stderr)
    subs = formatting.clean(parser.parse(srt_file))
    cur_sub = None
    final_subs = []

    for sub in subs:
        if cur_sub is None or cur_sub.lines[0] != sub.lines[0]:
            cur_sub = sub
            final_subs.append(cur_sub)
            cur_sub.index = len(final_subs)
        else:
            cur_sub.end = sub.end
            cur_sub.lines += sub.lines[1:]
    print('\tDone!', file=sys.stderr)
    return final_subs

def get_translation(oai_client, sub, in_file, addl_context):
    base = os.path.basename(in_file)
    out_file = f'/tmp/{base}.mp3'
    print(f'\tExtracting audio\t{sub.start} -> {sub.end}', file=sys.stderr)
    try:
        (st, st_err) = (
            ffmpeg
            .input(in_file)
            .output(out_file, acodec='libmp3lame', ac=1, ar='44100', audio_bitrate='128k', ss=datetime_to_seconds(sub.start), to=datetime_to_seconds(sub.end))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)
        exit(1)

    timeout = timeout_for_sub(sub)
    print(f'\tTranslating audio\t{sub.start} -> {sub.end}\t(Timeout {timeout} sec)', file=sys.stderr)
    with open(out_file, 'rb') as f:
        tr = oai_client.with_options(timeout=timeout).audio.translations.create(
            model = "whisper-1",
            file = f,
            prompt = addl_context,
            response_format = "text",
        )

    return subtitle.Subtitle(
        index = sub.index,
        start = sub.start,
        end = sub.end,
        lines = [speaker(sub), tr],
    )

def add_translations(video_file, srt_file):
    out_video = f'{video_file}.translated.mp4'

    # using subprocess here because wrangling the ffmpeg-python API for this is difficult
    subprocess.check_call(['ffmpeg', '-nostdin', '-y', '-i', video_file, '-i', srt_file, '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text', '-map', '0', '-map', '1', '-metadata:s:s:1', 'language=eng', out_video],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def datetime_to_seconds(dt: time) -> float:
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1000000

def format_time(t):
    if t.hour > 0:
        return f"{t.hour}:{t.minute:02d}:{t.second:02d}"
    else:
        return f"{t.minute}:{t.second:02d}"

def timeout_for_sub(sub) -> float:
    return min(MAX_TIMEOUT, (datetime_to_seconds(sub.end) - datetime_to_seconds(sub.start)) * TIMEOUT_RATIO)

def speaker(sub) -> str:
    return sub.lines[0]

def text(sub) -> str:
    return "\n".join(sub.lines[1:])

if __name__ == "__main__":
    main()