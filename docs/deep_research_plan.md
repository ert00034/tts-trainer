Below is a “local-first” recipe that people are already using on RTX 4090-class cards to give a Discord bot a convincing custom voice, transcribe replies in real time, and even tell speakers apart.

---

## 1 · Pick a voice-cloning TTS

| Model                    | Why it fits a 10 min dataset                                                                             | Quality / speed on 4090                                                                                       | Notes                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Coqui XTTS v2**        | Zero-shot voice-cloning from as little as \~6 s of reference audio – you don’t need to fine-tune at all. | 200 ms time-to-first-chunk, sustained real-time when streamed in 2-second blocks on a 4090. ([baseten.co][1]) | \~650 MB fp16; supports English out-of-the-box; simple API. |
| **Coqui VITS fine-tune** | Fine-tune the small VITS base for 1-2 epochs on your 10 min corpus for slightly sharper articulation.    | Real-time with 4090; training finishes in <30 min. ([github.com][2])                                          | Needs aligned text; use Whisper for transcripts.            |
| **Tortoise-TTS**         | Gives the “ultra-studio” sound if you can tolerate \~1 s latency.                                        | \~1 s per sentence on “ultra\_fast” preset with 4090. ([github.com][3])                                       | Not streamable; good fallback for off-line clips.           |

**Recommendation:** start with XTTS v2. You get instant voice-cloning, streaming support, and a simple upgrade path to VITS if you want to squeeze quality later.

---

## 2 · Prepare the 10 min reference audio

1. **Capture mono 24 kHz or 48 kHz WAV** (16-bit PCM).
2. **Trim silences** automatically with `ffmpeg -af silenceremove`.
3. **Denoise** lightly (e.g., `rnnoise` or Audacity’s noise-reduction). ([github.com][2])
4. **Loudness-normalize** to –23 LUFS (`ffmpeg -af loudnorm`).
5. Keep a single clean file (`ref.wav`); XTTS only needs that one file.

If you decide to fine-tune VITS later, also run Whisper to get an SRT transcript you can feed to the trainer.

---

## 3 · Streaming TTS with XTTS v2

```python
# pip install "TTS>=0.22"  torchaudio soundfile
from TTS.api import TTS
from pathlib import Path
tts = TTS(model_name="tts_models/en/xtts_v2").to("cuda")

def stream_chunks(text):
    for chunk, rate in tts.tts_with_preset(
            text,
            speaker_wav="ref.wav",
            stream_output=True,        # enables 2-s chunk generator
            split_sentences=False):
        yield chunk, rate             # 48000 Hz 16-bit PCM
```

The generator yields play-ready 48 kHz PCM buffers every \~200 ms on a 4090. ([baseten.co][1])

---

## 4 · Injecting audio into a Discord voice channel

```python
import discord, asyncio, subprocess

class PCMSource(discord.AudioSource):
    def __init__(self, pcm_iter):
        self.pcm_iter = pcm_iter
    def read(self):
        chunk = next(self.pcm_iter, None)
        return chunk[0] if chunk else b''           # 20 ms frames
    def is_opus(self): return False                # raw PCM

async def speak(vc, text):
    source = PCMSource(stream_chunks(text))
    vc.play(source)

client = discord.Client(intents=discord.Intents().all())

@client.event
async def on_message(msg):
    if msg.content.startswith('!say '):
        vc = await msg.author.voice.channel.connect()
        await speak(vc, msg.content[5:])

client.run("TOKEN")
```

Discord expects 16-bit 48 kHz stereo PCM or Opus. `discord.py`’s `FFmpegPCMAudio` helper can wrap the raw stream if you want Opus. ([discordpy.readthedocs.io][4], [docs.discord4py.dev][5])

---

## 5 · Real-time Speech-to-Text

### Faster-Whisper

```python
# pip install faster-whisper
from faster_whisper import WhisperModel
wmodel = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe(path):
    segs, _ = wmodel.transcribe(
        path, vad_filter=True, beam_size=1, chunk_length=15)
    return " ".join(s.text for s in segs)
```

Faster-Whisper is 4× faster than the original and runs easily in real time on GPU with VAD enabled. ([github.com][6])

### Speaker identification (optional)

```python
# pip install -U pyannote.audio
from pyannote.audio import Pipeline
spk = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = spk("user_audio.wav")   # 16 kHz mono
```

`pyannote.audio` gives you per-speaker time-stamps that you can match to Discord user IDs by measuring voiceprint similarity over time. ([github.com][7], [huggingface.co][8])

---

## 6 · Putting it together

1. **Capture** raw Opus packets from Discord users (the fork [`discord-ext-audiorec`](https://github.com/.../) makes this easy) and decode to 16 kHz WAV.
2. **Transcribe** with Faster-Whisper; optionally run pyannote for diarization.
3. **Generate reply text** (your existing bot logic / LLM).
4. **Stream reply audio** with XTTS as shown.
5. **Loop**.

On a single 4090 you can comfortably run Faster-Whisper (`--compute_type float16`) and one XTTS stream simultaneously; peak VRAM is \~7 GB total. Use asyncio queues so STT and TTS don’t block each other.

---

### Summary

* **Start with Coqui XTTS v2** for zero-shot voice-clone quality that already streams in real time on your 4090.
* **Clean your 10 min reference audio** once; no further training required unless you want to fine-tune VITS for extra polish.
* **Use Faster-Whisper** (+VAD) for low-latency transcription, and **pyannote** if you need speaker separation.
* **Pump PCM directly into discord.py’s voice client** or through FFmpeg for Opus.
  With those pieces you’ll have a fully-local, privacy-friendly Discord bot that talks back in your custom character voice and understands the channel in real time.

[1]: https://www.baseten.co/blog/streaming-real-time-text-to-speech-with-xtts-v2/?utm_source=chatgpt.com "Streaming real-time text to speech with XTTS V2 | Baseten Blog"
[2]: https://github.com/coqui-ai/TTS/discussions/2507?utm_source=chatgpt.com "Best Procedure For Voice Cloning - My Experience So Far #2507"
[3]: https://github.com/neonbjb/tortoise-tts/issues/574?utm_source=chatgpt.com "Absolute fastest inference speed · Issue #574 · neonbjb/tortoise-tts"
[4]: https://discordpy.readthedocs.io/en/latest/api.html?highlight=ffmpeg&utm_source=chatgpt.com "API Reference - Discord.py"
[5]: https://docs.discord4py.dev/en/developer/api/voice.html?utm_source=chatgpt.com "Voice Related - discord.py-message-components"
[6]: https://github.com/SYSTRAN/faster-whisper?utm_source=chatgpt.com "Faster Whisper transcription with CTranslate2 - GitHub"
[7]: https://github.com/pyannote/pyannote-audio?utm_source=chatgpt.com "pyannote/pyannote-audio: Neural building blocks for speaker ..."
[8]: https://huggingface.co/pyannote/speaker-diarization-3.1?utm_source=chatgpt.com "pyannote/speaker-diarization-3.1 - Hugging Face"
