from eval.models import Word, TranscriptResult, Reference, Window, NormalizedAudio, Chunk, EngineScore

def test_word_roundtrips_through_dict():
    w = Word(text="hello", start=0.0, end=0.3, speaker="SPEAKER_00")
    assert Word.from_dict(w.to_dict()) == w

def test_word_rounds_timestamps_on_serialize():
    w = Word(text="hi", start=0.123456, end=0.98765)
    assert w.to_dict()["start"] == 0.123
    assert w.to_dict()["end"] == 0.988

def test_transcript_result_roundtrips_with_words():
    tr = TranscriptResult(
        engine_id="groq/whisper-large-v3-turbo",
        text="hello world",
        words=[Word("hello", 0.0, 0.3), Word("world", 0.4, 0.8)],
        meta={"rtf": 0.12},
    )
    back = TranscriptResult.from_dict(tr.to_dict())
    assert back == tr
    assert back.speakers is None

def test_reference_roundtrips_nested_windows():
    ref = Reference(
        audio_id="roncesvalles",
        windows=[Window(0.0, 180.0, [Word("hi", 1.0, 1.2, "SPEAKER_00")], corrected=True)],
    )
    assert Reference.from_dict(ref.to_dict()) == ref

def test_normalized_audio_roundtrips():
    na = NormalizedAudio(wav_path="/tmp/n.wav", duration=12.5)
    assert NormalizedAudio.from_dict(na.to_dict()) == na

def test_chunk_roundtrips_and_rounds():
    c = Chunk(path="/tmp/c.wav", start=1.0, end=5.5)
    assert Chunk.from_dict(c.to_dict()) == c
    assert Chunk(path="x", start=1.23456, end=2.0).to_dict()["start"] == 1.235

def test_engine_score_roundtrips_with_none_fields():
    s = EngineScore(engine_id="e", cpwer=0.25, wer=0.1, cer=None)
    assert EngineScore.from_dict(s.to_dict()) == s

def test_engine_score_rounds_metric_floats():
    s = EngineScore(engine_id="e", wer=0.253333)
    assert s.to_dict()["wer"] == 0.253
