import random

from server.output_parser import ReplyParser, parse_full

REPLY = (
    '<say lang="fr">Presque ! On dit <en>I went to the cinema</en>, pas « I goed ». '
    "<en>What did you watch?</en></say>\n"
    '<fix>[{"type": "conjugation", "original": "I goed", "corrected": "I went", '
    '"rule_key": "past_simple_irregular", "explain_fr": "go est irrégulier"}]</fix>'
)


def stream(text: str, sizes: list[int]) -> tuple[list, str, ReplyParser]:
    parser = ReplyParser()
    segments, display, i = [], "", 0
    for n in sizes:
        segs, disp = parser.feed(text[i: i + n])
        segments += segs
        display += disp
        i += n
    segs, disp = parser.feed(text[i:])
    segments += segs
    display += disp
    segs, disp = parser.close()
    return segments + segs, display + disp, parser


def test_segments_split_by_language_and_clause():
    reply = parse_full(REPLY)
    assert [(s.lang, s.text) for s in reply.segments] == [
        ("fr", "Presque !"),
        ("fr", "On dit"),
        ("en", "I went to the cinema"),
        ("fr", "pas « I goed »."),
        ("en", "What did you watch?"),
    ]
    assert reply.fixes == [{
        "type": "conjugation", "original": "I goed", "corrected": "I went",
        "rule_key": "past_simple_irregular", "explain_fr": "go est irrégulier",
    }]
    assert "<" not in reply.say_text and "I went to the cinema" in reply.say_text


def test_same_result_whatever_the_token_boundaries():
    reference = parse_full(REPLY)
    rng = random.Random(0)
    for _ in range(200):
        sizes = [rng.randint(1, 6) for _ in range(len(REPLY))]
        segments, display, parser = stream(REPLY, sizes)
        assert [(s.lang, s.text) for s in segments] == [(s.lang, s.text) for s in reference.segments]
        assert parser.reply.fixes == reference.fixes
        assert display.strip() == reference.say_text


def test_first_segment_released_before_end_of_sentence():
    parser = ReplyParser()
    segs, _ = parser.feed('<say lang="en">Nice try, but we usually say')
    assert [s.text for s in segs] == ["Nice try,"]


def test_missing_tags_are_tolerated():
    reply = parse_full("Great job! Tell me more about your weekend.")
    assert [s.text for s in reply.segments] == ["Great job!", "Tell me more about your weekend."]
    assert reply.fixes == [] and not reply.fix_parse_error


def test_broken_fix_json_is_reported_not_raised():
    reply = parse_full('<say lang="en">Good.</say><fix>[{"original": "x", </fix>')
    assert reply.fixes == [] and reply.fix_parse_error


def test_unescaped_quotes_in_explanation_are_repaired():
    reply = parse_full(
        '<say lang="en">We say depends on.</say>\n<fix>[{"type": "preposition", "original": "depends of", '
        '"corrected": "depends on", "rule_key": "depend_on", "explain_fr": "On dit toujours "depend on" en anglais."}, '
        '{"type": "grammar", "original": "he have", "corrected": "he has", "rule_key": "third_person_s", '
        '"explain_fr": "3e personne : "has"."}]</fix>'
    )
    assert not reply.fix_parse_error
    assert [(f["original"], f["corrected"]) for f in reply.fixes] == [("depends of", "depends on"), ("he have", "he has")]
    assert reply.fixes[0]["explain_fr"] == 'On dit toujours "depend on" en anglais.'


def test_untagged_english_sentence_in_french_reply_is_spoken_in_english():
    reply = parse_full('<say lang="fr">On dit <en>I can\'t wait</en>. Now, let\'s try to speak a little English! '
                       'How are you doing today?</say><fix>[]</fix>')
    assert [(s.lang, s.text) for s in reply.segments] == [
        ("fr", "On dit"), ("en", "I can't wait"), ("en", "Now, let's try to speak a little English!"),
        ("en", "How are you doing today?"),
    ]


def test_missing_closing_bracket_is_tolerated():
    reply = parse_full('<say>Ok.</say><fix>[{"type": "word_order", "original": "I want that you come", '
                       '"corrected": "I want you to come", "rule_key": "want_to", "explain_fr": "Après "want"."}</fix>')
    assert reply.fixes[0]["corrected"] == "I want you to come" and not reply.fix_parse_error


def test_unknown_fix_type_falls_back_to_grammar():
    reply = parse_full('<say>Ok.</say><fix>[{"type": "weird", "original": "a", "corrected": "b"}]</fix>')
    assert reply.fixes[0]["type"] == "grammar" and reply.fixes[0]["rule_key"] == "weird"
