import subprocess
import numpy as np


def start_audio_capture(audio_queue):
    device = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

    cmd = [
        "ffmpeg",
        "-f", "pulse",
        "-i", device,
        "-ac", "1",
        "-ar", "16000",
        "-f", "s16le",
        "-"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

    chunk_size = 16000 * 2 * 1  # 1 second of audio (16000 samples * 2 bytes/sample * 1 channel)

    while True:
        data = process.stdout.read(chunk_size)
        if not data:
            print("❌ No audio data")
            continue
        print("✅ Audio chunk received:", len(data))
        audio = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        audio_queue.put(audio)
