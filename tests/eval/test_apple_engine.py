from eval.engines.apple_voicememos import parse_apple_transcript

SAMPLE = """00:00:00 Speaker 1
Hello there everyone.
00:00:04 Speaker 2
General Kenobi.
"""

def test_parses_speakers_and_words():
    tr = parse_apple_transcript(SAMPLE)
    assert tr.engine_id == "apple/voicememos"
    assert tr.words[0].text == "Hello"
    assert tr.words[0].speaker == "SPEAKER_00"
    assert any(w.speaker == "SPEAKER_01" and w.text == "General" for w in tr.words)

def test_word_times_are_monotonic():
    tr = parse_apple_transcript(SAMPLE)
    times = [w.start for w in tr.words]
    assert times == sorted(times)
