import argparse
import os
import re
import sys

import ffmpeg

def main():
    parser = argparse.ArgumentParser(description="Transcribe and translate meeting recordings. Requires ffmpeg to be installed. Requires a whisperapi.com key in $WHISPERAPI_KEY.")
    
    parser.add_argument("input_file", help="Input file to transcribe", type=str)
    parser.add_argument("-l", "--language", help="Original language of the meeting audio", type=str, default="pt")
    parser.add_argument("-n", "--num_speakers", help="Number of speakers, automatically determined from subtitled files if not set", type=int)
    parser.add_argument("-c", "--context", help="Freetext context to help the transcription, useful for teaching phrases/acronyms", type=str, default="")

    args = parser.parse_args()

    key = os.environ.get("WHISPERAPI_KEY")
    if key is None:
        print("Please set a whisperapi.com key in $WHISPERAPI_KEY")
        exit(1)

    if not os.path.isfile(args.input_file):
        print("Input file does not exist")
        exit(1)

    num_speakers = args.num_speakers or get_num_speakers(args.input_file)

    audio_file = get_audio_file(args.input_file)

    headers = {
        'Authorization': f'Bearer {key}'
    }

    with open(audio_file, "rb") as f:
        req = {
             "fileType": args.input_file.split(".")[-1],
              "diarization": "true",
              "initialPrompt": args.context,
              "language": args.language,
              "task": "translate",
        }
        if num_speakers is not None:
            req["numSpeakers"] = num_speakers

        print('\nCalling translation service...', file=sys.stderr)
        resp = requests.post("https://transcribe.whisperapi.com", headers=headers, files={'file': f}, data=req)
        print('\tDone!', file=sys.stderr)
        print(response.text)
            

def get_num_speakers(in_file: str) -> int:
    (st, st_err) = (
        ffmpeg
        .input(in_file)
        .output('pipe:', format='srt')
        .run(capture_stdout=True, capture_stderr=True)
    )

    speaker_pattern = r'\((.*?)\)'
    speakers = set(re.findall(speaker_pattern, st.decode("utf-8")))

    print(f'\nFound {len(speakers)} speakers:', file=sys.stderr)
    for speaker in speakers:
        print(f'\t{speaker}', file=sys.stderr)
    return len(speakers)
    
def get_audio_file(in_file: str) -> str:
    base = os.path.basename(in_file)
    out_file = f'/tmp/{base}.mp3'
    print('\nExtracting and converting audio', file=sys.stderr)
    (st, st_err) = (
        ffmpeg
        .input(in_file)
        .output(out_file, acodec='libmp3lame', ac=1, ar='44100', audio_bitrate='96k')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    print('\tDone!', file=sys.stderr)
    return out_file

if __name__ == "__main__":
    main()