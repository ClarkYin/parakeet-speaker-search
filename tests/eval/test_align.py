from eval.align import align

def test_equal_tokens():
    ops = align("the cat sat", "the cat sat")
    assert [o["op"] for o in ops] == ["equal", "equal", "equal"]

def test_substitution_marked():
    ops = align("the cat sat", "the dog sat")
    sub = [o for o in ops if o["op"] == "sub"]
    assert sub == [{"op": "sub", "ref": "cat", "hyp": "dog"}]

def test_insertion_and_deletion():
    ins = [o for o in align("a c", "a b c") if o["op"] == "ins"]
    assert ins == [{"op": "ins", "ref": None, "hyp": "b"}]
    dele = [o for o in align("a b c", "a c") if o["op"] == "del"]
    assert dele == [{"op": "del", "ref": "b", "hyp": None}]
