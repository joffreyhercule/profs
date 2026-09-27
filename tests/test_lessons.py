"""Leçons : programme écrit, plan généré par le LLM, suivi de l'avancement par les balises <lesson>."""

import json

import pytest

from server.lessons import Lesson, LessonState, parse_generated
from server.subjects import load_subjects

BOTANIQUE = load_subjects()["botanique"]
PLAN = {"title": "Les champignons", "summary": "Ce que sont les champignons.",
        "sections": [{"title": f"Partie {i}", "points": ["Une notion."], "check": "Compris ?"} for i in range(1, 4)],
        "quiz": [{"question": f"Question {i} ?", "answer": f"Réponse {i}."} for i in range(1, 5)]}


def test_program_lessons_are_complete_and_in_order():
    keys = [lesson.key for lesson in BOTANIQUE.lessons]
    assert keys == ["programme:01-organes", "programme:02-photosynthese", "programme:03-seves"]
    for lesson in BOTANIQUE.lessons:
        assert len(lesson.sections) == 6 and len(lesson.quiz) == 10
        assert all(len(s["points"]) >= 5 and s["check"] for s in lesson.sections)
        assert all(q["answer"] for q in lesson.quiz)


def test_lesson_plan_goes_into_the_teacher_prompt_only_for_a_lesson():
    lesson = BOTANIQUE.lessons[2]
    prompt = BOTANIQUE.system_prompt("", "Léa", lesson)
    assert "# Plan de la leçon : La circulation de la sève" in prompt
    assert "## Partie 3 : Ce qui fait monter la sève brute" in prompt
    assert "(réponse attendue : Le cambium.)" in prompt and "<lesson>" in prompt
    assert prompt.rstrip().endswith("premier cours ensemble. Présente-toi en une phrase, puis évalue ce que "
                                    "l'élève sait déjà avec une question simple.")
    assert "Plan de la leçon" not in BOTANIQUE.system_prompt("", "Léa")


def test_generated_plan_is_read_even_inside_a_code_fence():
    lesson = Lesson.from_plan("gen:1", parse_generated("Voici :\n```json\n" + json.dumps(PLAN) + "\n```"))
    assert lesson.title == "Les champignons" and len(lesson.sections) == 3 and len(lesson.quiz) == 4
    with pytest.raises(ValueError):
        Lesson.from_plan("gen:2", {"title": "Vide", "sections": [], "quiz": []})


def test_progress_follows_the_teacher_through_lesson_and_quiz():
    state = LessonState(Lesson.from_plan("gen:1", PLAN))
    assert state.apply({"section": 1}) is False  # déjà là
    assert state.apply({"section": 2}) and state.tag(7) == "[Leçon : partie 2/3 · 7 min]"
    state.apply({"section": 1})  # retour en arrière pour réexpliquer : la partie la plus avancée reste 2
    assert (state.section, state.reached) == (1, 2)
    state.apply({"question": 1})
    state.apply({"question": 2, "correct": True})
    state.apply({"question": 3, "correct": False})
    assert state.results == {1: True, 2: False} and state.reached == 3
    assert state.tag(21) == "[Quiz : question 3/4 posée · 1 bonne(s) réponse(s) sur 2 · 21 min]"
    state.apply({"section": 1})  # ignoré pendant le quiz
    assert state.question == 3
    state.apply({"question": 4, "correct": True})
    state.apply({"done": True, "correct": True})
    assert state.done and state.score == 3 and state.progress()["results"] == {"1": True, "2": False, "3": True,
                                                                               "4": True}
    assert state.tag(29) == "[Quiz terminé : 3/4 · 29 min]"


def test_interrupted_lesson_resumes_where_it_stopped():
    lesson = Lesson.from_plan("gen:1", PLAN)
    state = LessonState(lesson)
    assert state.resume_point() is None
    state.apply({"section": 3})
    resumed = LessonState.resumed(lesson, state.row())
    assert resumed.section == 3 and resumed.resume_point() == "la partie 3 (Partie 3)"
    state.apply({"question": 2})
    resumed = LessonState.resumed(lesson, {**state.row()})
    assert resumed.question == 2 and resumed.resume_point() == "la question 2 du quiz"


def test_quiz_verdict_names_the_question_it_grades():
    """Le prof note explicitement la question à laquelle l'élève vient de répondre : pas d'ambiguïté avec
    la question qu'il pose dans la même réponse."""
    state = LessonState(Lesson.from_plan("gen:1", PLAN))
    state.apply({"question": 1})
    state.apply({"answered": 1, "right": False, "question": 2})
    state.apply({"answered": 2, "right": True, "question": 3})
    state.apply({"answered": 3, "right": True, "question": 4})
    state.apply({"answered": 4, "right": True, "done": True})
    assert state.results == {1: False, 2: True, 3: True, 4: True} and state.done and state.score == 3


def test_nested_marker_still_starts_the_quiz():
    """Vu en vrai : au passage au quiz, Gemma a écrit {"section": {"question": 1}}."""
    state = LessonState(Lesson.from_plan("gen:1", PLAN))
    state.apply({"section": 3})
    assert state.apply({"section": {"question": 1}}) and state.question == 1
    state.apply({"section": {"answered": 1, "right": True, "question": 2}})
    assert state.results == {1: True} and state.question == 2


def test_what_the_teacher_says_fixes_a_wrong_quiz_number():
    """Vu en vrai : « Question 1 : … » annoncé avec la balise {"section": 1}."""
    state = LessonState(Lesson.from_plan("gen:1", PLAN))
    state.apply({"section": 3})
    state.apply({"section": 1}, "Place au quiz ! Question 1 : pourquoi ?")
    assert (state.question, state.section) == (1, 3)
    state.apply({"answered": 1, "right": True}, "Bravo ! Question 2 : et ensuite ?")
    assert state.question == 2 and state.results == {1: True}
    state.apply({"section": 2}, "Revenons à la partie 2.")  # hors quiz et avant la fin : la balise fait foi
    assert state.question == 2


def test_sound_alike_words_are_spotted_but_real_confusions_are_not():
    from server.lessons import homophones, sound

    assert sound("poêles") == sound("poils") and sound("chaîne") == sound("chêne")
    assert sound("sépales") != sound("pétales") and sound("xylème") != sound("phloème")
    assert homophones("Des poêles absorbants.", "Les poils absorbants.") == [("poêles", "poils")]
    assert homophones("Les pétales", "Les sépales") == []
