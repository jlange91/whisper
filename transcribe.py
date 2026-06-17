#!/usr/bin/env python3
"""Real-time microphone transcription using mlx-whisper with sliding window."""

import argparse
import collections
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import mlx_whisper

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.03
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


class Transcriber:
    def __init__(self, model, language, silence_threshold, silence_duration, interval, timestamps):
        print(f"Chargement du modèle '{model}'...", end=" ", flush=True)
        self._model = model
        self._language = language
        self._timestamps = timestamps
        self._silence_threshold = silence_threshold
        self._silence_frames = int(silence_duration / CHUNK_DURATION)
        self._interval = interval

        # Warm up: forces model download + JIT compile before first real use
        mlx_whisper.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=model,
            verbose=None,
        )
        print("prêt.")

        self._audio_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        # Single-slot pending transcription request: (audio_array, is_final)
        # Latest request always overwrites previous unprocessed one.
        self._pending_lock = threading.Lock()
        self._pending: tuple | None = None
        self._pending_event = threading.Event()

        self._history: collections.deque = collections.deque()  # (time_str, text)
        self._live_text: str = ""
        self._is_processing: bool = False
        self._display_lock = threading.Lock()
        self._console = Console()
        self._layout: Layout | None = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self._console.print(f"\n[audio status: {status}]", style="red", highlight=False)
        self._audio_q.put(indata.copy())

    def _submit(self, buffer: list, is_final: bool):
        audio = np.concatenate(buffer).flatten().astype(np.float32)
        with self._pending_lock:
            self._pending = (audio, is_final)
        self._pending_event.set()

    def _transcribe(self, audio: np.ndarray) -> str:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
            language=self._language,
            condition_on_previous_text=False,
            verbose=None,
        )
        if self._timestamps:
            return " ".join(
                f"[{s['start']:.1f}s] {s['text'].strip()}"
                for s in result.get("segments", [])
            ).strip()
        return result.get("text", "").strip()

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="history"),
            Layout(name="live", size=5),
        )
        return layout

    def _render_header(self) -> Panel:
        if self._is_processing:
            status = Text("⟳  transcription...", style="bold yellow")
        else:
            status = Text("●  écoute...", style="bold green")
        return Panel(
            status,
            title="[bold]🎙  Whisper Live[/bold]",
            subtitle=f"[dim]{self._model}[/dim]",
            border_style="dim",
        )

    def _render_history(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        for ts, text in self._history:
            table.add_row(ts, text)
        return Panel(table, title="[bold]Historique[/bold]", border_style="blue")

    def _render_live(self) -> Panel:
        if self._live_text:
            content = Text(self._live_text, style="bold")
        else:
            content = Text("en attente...", style="dim italic")
        return Panel(content, title="[bold]En cours[/bold]", border_style="yellow")

    def _refresh_display(self):
        if self._layout is None:
            return
        with self._display_lock:
            self._layout["header"].update(self._render_header())
            self._layout["history"].update(self._render_history())
            self._layout["live"].update(self._render_live())

    def _worker(self):
        """Dedicated thread: pulls pending requests and runs inference."""
        while not self._stop.is_set():
            if not self._pending_event.wait(timeout=0.1):
                continue
            self._pending_event.clear()

            with self._pending_lock:
                if self._pending is None:
                    continue
                audio, is_final = self._pending
                self._pending = None

            self._is_processing = True
            self._refresh_display()

            text = self._transcribe(audio)

            self._is_processing = False
            if text:
                if is_final:
                    ts = time.strftime("%H:%M:%S")
                    self._history.append((ts, text))
                    self._live_text = ""
                else:
                    self._live_text = text
            self._refresh_display()

    def _vad_loop(self):
        """VAD + sliding window: re-transcribes buffer every --interval seconds."""
        speech_buffer: list = []
        silence_count = 0
        is_speaking = False
        last_transcribe = 0.0

        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

        while not self._stop.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            now = time.monotonic()

            if rms > self._silence_threshold:
                speech_buffer.append(chunk)
                silence_count = 0
                if not is_speaking:
                    is_speaking = True
                    last_transcribe = now

                # Periodic re-transcription while speaking
                if now - last_transcribe >= self._interval:
                    last_transcribe = now
                    self._submit(speech_buffer, is_final=False)
            else:
                if is_speaking:
                    speech_buffer.append(chunk)
                    silence_count += 1
                    if silence_count >= self._silence_frames:
                        self._submit(speech_buffer, is_final=True)
                        speech_buffer = []
                        silence_count = 0
                        is_speaking = False

    def run(self):
        self._layout = self._make_layout()
        self._refresh_display()
        try:
            with Live(
                self._layout,
                console=self._console,
                refresh_per_second=8,
                screen=True,
            ):
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    blocksize=CHUNK_SIZE,
                    dtype="float32",
                    callback=self._audio_callback,
                ):
                    self._vad_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            self._console.print("\n[arrêt]")


def main():
    parser = argparse.ArgumentParser(
        description="Transcription microphone en temps réel via mlx-whisper"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-small-mlx",
        help="Repo HuggingFace du modèle (défaut: mlx-community/whisper-small-mlx)",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="CODE",
        help="Code langue ISO 639-1 (ex: fr, en). Omis = auto-détection",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Seuil RMS de détection de parole (défaut: 0.01)",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=0.8,
        metavar="SECONDES",
        help="Durée de silence avant finalisation (défaut: 0.8s)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="SECONDES",
        help="Intervalle de re-transcription pendant la parole (défaut: 1.0s)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Afficher les timestamps de chaque segment",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Lister les périphériques audio disponibles et quitter",
    )

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    Transcriber(
        model=args.model,
        language=args.language,
        silence_threshold=args.threshold,
        silence_duration=args.silence,
        interval=args.interval,
        timestamps=args.timestamps,
    ).run()


if __name__ == "__main__":
    main()
