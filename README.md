# Video Subtitle Generator

This project generates real-time subtitles for videos that do not have subtitles by capturing system audio, transcribing speech using Whisper, and displaying subtitles as an on-screen overlay.

It works for:

- YouTube videos
- Movies
- Online courses
- Video players
- Any system audio

## Features

- Captures system audio using FFmpeg
- Transcribes speech using Faster-Whisper
- Translates speech to English
- Displays live subtitle overlay
- Runs fully offline
- Multi-threaded pipeline

## Project Structure

```bash
project/
│
├── app.py
├── audio_capture.py
├── transcriber.py
├── overlay.py
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

### Files Overview

| File              | Purpose                               |
|-------------------|---------------------------------------|
|app.py             | Main entry point, starts threads      |
|audio_capture.py   | Captures system audio using FFmpeg    |
|transcriber.py     | Converts audio → text using Whisper   |
|overlay.py         | Displays subtitles overlay window     |
|models/            | Whisper model files                   |

## How It Works

### Pipeline

```bash
System Audio
     ↓
FFmpeg Audio Capture
     ↓
Audio Queue
     ↓
Whisper Transcription
     ↓
Text Queue
     ↓
Subtitle Overlay Window
```

The app uses two background threads:

1. Audio capture thread
2. Transcription thread

The main thread runs the subtitle overlay UI.

## Requirements

Install dependencies:

```bash
pip install faster-whisper customtkinter numpy
```

Install FFmpeg:

- Windows: Download from <https://ffmpeg.org/download.html>
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

Make sure PulseAudio monitor device exists:

```bash
pactl list sources | grep monitor
```

Then update device name in: `audio_capture.py`

## Running the Application

```bash
python app.py
```

Then:

1. Play a video
2. Subtitles will appear at the bottom of the screen
3. Overlay stays on top of all windows

> [!NOTE]
> *Known issue:* The audio chunk capturing, and translation take time, so the subtitles appear with adelay, making the video and subtitles out of sync.
