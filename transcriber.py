from faster_whisper import WhisperModel
import numpy as np

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    download_root="models",
    local_files_only=True,
    cpu_threads=4,
    num_workers=4
)

def start_transcription(audio_queue, text_queue):
    buffer = np.array([], dtype=np.float32)

    while True:
        audio = audio_queue.get()
        buffer = np.concatenate((buffer, audio))

        # Wait until we have ~10 seconds audio
        if len(buffer) < 16000 * 10:
            continue

        print("🧠 Transcribing 10 sec audio...")

        segments, _ = model.transcribe(
            buffer,
            task="translate",
            beam_size=1,
            vad_filter=True
        )

        texts = [seg.text for seg in segments]
        print("🧠 Segments:", texts)

        text = " ".join(texts)

        if text.strip():
            print("✅ Final text:", text)
            text_queue.put(text)

        # Clear buffer after transcription
        buffer = np.array([], dtype=np.float32)
