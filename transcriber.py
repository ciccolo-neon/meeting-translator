import argparse
from datetime import time
import os
import re
import sys
import threading

import ffmpeg
from openai import OpenAI
from pysubparser import parser
from pysubparser.cleaners import formatting

MAX_TRIES = 3
TIMEOUT_RATIO = 2.0  # give the translation call twice as long as the audio length
MAX_TIMEOUT = 30.0 # but don't let it go over 30 seconds


def main():
    parser = argparse.ArgumentParser(description="Transcribe and translate meeting recordings. Requires ffmpeg to be installed. Requires a whisperapi.com key in $WHISPERAPI_KEY.")
    
    parser.add_argument("input_file", help="Input file to transcribe", type=str)
    parser.add_argument("output_file", help="Location of output transcript", type=str)
    parser.add_argument("-c", "--context", help="Freetext context to help the transcription, useful for teaching phrases/acronyms", type=str, default="")

    args = parser.parse_args()

    key = os.environ.get("WHISPERAPI_KEY")
    if key is None:
        print("Please set a whisperapi.com key in $WHISPERAPI_KEY")
        exit(1)

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
    with open(args.output_file, 'w') as f:
        for sub in proc_subs:
            text = get_translation(oai_client, sub, args.input_file, args.context)
            f.write(f'({sub['speaker']})\t{format_time(sub['start'])}\n{text}\n')
            f.flush() # flush to file so we get something if the call crashes/hangs
    print('\tDone!', file=sys.stderr)
        

def process_subtitles(srt_file: str) -> list:
    print('\nProcessing subtitles', file=sys.stderr)
    subs = formatting.clean(parser.parse(srt_file))
    cur_sub = None
    final_subs = []
    speaker_pattern = re.compile(r'^\((.+)\) ')

    for sub in subs:
        m = re.match(speaker_pattern, sub.text)
        speaker = m.group(1)
        mysub = {
            'text': re.sub(speaker_pattern, '', sub.text),
            'start': sub.start,
            'end': sub.end,
            'speaker': speaker 
        }
        if cur_sub is None or cur_sub['speaker'] != speaker:
            cur_sub = mysub
            final_subs.append(cur_sub)
        else:
            cur_sub['end'] = mysub['end']        
            cur_sub['text'] += ' ' + mysub['text']

    print('\tDone!', file=sys.stderr)
    return final_subs

def get_translation(oai_client, sub, in_file, context) -> str:
    base = os.path.basename(in_file)
    out_file = f'/tmp/{base}.mp3'
    print(f'\tExtracting audio\t{sub['start']} -> {sub['end']}', file=sys.stderr)
    try:
        (st, st_err) = (
            ffmpeg
            .input(in_file)
            .output(out_file, acodec='libmp3lame', ac=1, ar='44100', audio_bitrate='128k', ss=datetime_to_seconds(sub['start']), to=datetime_to_seconds(sub['end']))
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)
        exit(1)

    timeout = timeout_for_sub(sub)
    print(f'\tTranslating audio\t{sub['start']} -> {sub['end']}\t(Timeout {timeout} sec)', file=sys.stderr)
    with open(out_file, 'rb') as f:
        tr = oai_client.with_options(timeout=timeout).audio.translations.create(
            model = "whisper-1",
            file = f,
            prompt = context,
            response_format = "text",
        )

    print(f'({sub['speaker']}) {sub['start']} -> {sub['end']}\n{sub['text']}->\n{tr}', file=sys.stderr)
    return tr

def datetime_to_seconds(dt: time) -> float:
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1000000

def format_time(t):
    if t.hour > 0:
        return f"{t.hour}:{t.minute:02d}:{t.second:02d}"
    else:
        return f"{t.minute}:{t.second:02d}"

def timeout_for_sub(sub) -> float:
    return min(MAX_TIMEOUT, (datetime_to_seconds(sub['end']) - datetime_to_seconds(sub['start'])) * TIMEOUT_RATIO)

if __name__ == "__main__":
    main()