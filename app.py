import threading
import queue

from audio_capture import start_audio_capture
from transcriber import start_transcription
from overlay import SubtitleOverlay

audio_queue = queue.Queue()
text_queue = queue.Queue()


def start_background():
    threading.Thread(target=start_audio_capture, args=(audio_queue,), daemon=True).start()
    threading.Thread(target=start_transcription, args=(audio_queue, text_queue), daemon=True).start()


def start_overlay():
    overlay = SubtitleOverlay(text_queue)
    overlay.run()


if __name__ == "__main__":
    start_background()
    start_overlay()
