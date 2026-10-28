import pytest

from app.models import DEMO_NOTICE, ValidationError
from app.workflow import run_workflow, safe_filename, validate_inputs


def test_empty_topic_rejected():
    with pytest.raises(ValidationError):
        validate_inputs("   ")


def test_symbols_only_topic_rejected():
    with pytest.raises(ValidationError):
        validate_inputs("!!!???")


def test_invisible_bom_character_removed():
    assert validate_inputs("﻿Anatomy")[0] == "Anatomy"


def test_keyword_defaults_to_topic():
    assert validate_inputs("USMLE Step 1 study plan", "") == (
        "USMLE Step 1 study plan",
        "USMLE Step 1 study plan",
    )


def test_workflow_produces_structured_draft(tmp_path):
    result = run_workflow("USMLE Step 1 study plan", "step 1 study plan", output_dir=tmp_path)
    article = result.article
    assert article.title
    assert article.meta_description and len(article.meta_description) <= 160
    assert len(article.sections) >= 3
    assert article.faq
    assert result.checks_passed
    assert DEMO_NOTICE in article.to_markdown()


def test_output_file_created(tmp_path):
    result = run_workflow("Anatomy revision tips", output_dir=tmp_path)
    text = open(result.output_path, encoding="utf-8").read()
    assert result.output_path.endswith(".md")
    assert text.startswith("# ")
    assert "## Frequently Asked Questions" in text
    assert DEMO_NOTICE in text


def test_safe_filename_strips_dangerous_characters():
    name = safe_filename('../..\\evil:"name"?*')
    assert "/" not in name and "\\" not in name and ":" not in name
    assert name.endswith(".md")


def test_unwritable_output_dir_gives_clear_error(tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(OSError, match="Could not save"):
        run_workflow("Anatomy", output_dir=blocker / "sub")
