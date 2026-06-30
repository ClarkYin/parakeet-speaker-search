# eval — transcription evaluation harness

Run multiple ASR/diarization engines on an audio file, build a reference,
and score each engine by cpWER/WER/CER/DER.

## Quickstart

    python -m pip install -e ".[dev,engines]"
    python -m eval run path/to/audio.m4a --work-dir runs/roncesvalles

Engines with missing API keys are skipped automatically. Available without keys:
Parakeet (local MLX). With keys in `.env`: Groq Whisper, Deepgram. Optional:
AssemblyAI, ElevenLabs, OpenAI, Gemini.

## Reference correction

The first run writes `runs/<name>/reference.json` with up to 5 sampled ~3-min
windows, bootstrapped by ROVER consensus. Edit the `words` and set
`"corrected": true` per window, then re-run to get absolute scores. Provide
Apple's exported transcript via `--apple-transcript file.txt` to seed it.

## Output

`runs/<name>/report.md`, `report.html`, `scores.json`.
