# Overview

A tool for translating meeting audio into English-language subtitles, using the OpenAI Whisper API. 

# Outputs
- A subtitle (.srt) file, in the same directory as the input video (in.mp4 -> in.mp4.srt). SRT files are human readable, so this can be used as a transcript.
- A new video file with English softsub captions, in the same directory as the input video (in.mp4 -> in.mp4.translated.mp4). Any original subtitle tracks will be preserved.

# Usage

## Prerequisites
- This tool is intended for recorded meetings from Google Meet. It may work on other inputs, but no guarantees.
- The recording MUST have captions turned on (they are used for backdoor diarization).
- You need to have `ffmpeg` installed locally. (On mac, you can `brew install ffmpeg`).
- You need an OpenAI API key, which should be in your env as `OPENAI_API_KEY`.

## Invocation
`python3 ./transcriber.py ~/in.mp4`

You can use the `--context` flag if you want to add context, which can be useful for acronyms, terms of art, etc.

## Costs
For a 15-minute meeting, the tool took 90 seconds and cost 13 cents.