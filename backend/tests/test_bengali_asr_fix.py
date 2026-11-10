"""Bengali ASR post-correction tests."""

from __future__ import annotations

from app.language.bengali_asr_fix import fix_bengali_asr


def test_common_phonetic_repairs():
    assert "কথা" in fix_bengali_asr("তুমি কি আমার কছা বুঝতে পারো?")
    assert "মন ভালো" in fix_bengali_asr("আমার আজকে মুনভারো নেই")
    assert "ভাত" in fix_bengali_asr("তুমি কি আজকে ভাগ খেয়েছো?")
    assert "অনেক কিছু" in fix_bengali_asr("হ্যা, আনেকিছু নিয়ে বলতে পরাও।")
    assert "পারো" in fix_bengali_asr("বলতে পরাও")
    assert "আজকে" in fix_bengali_asr("আমি আতিকে অনেকিছু খেয়েছি।")


def test_orphan_kar_and_dup_token():
    assert fix_bengali_asr("া মার নাম কি").startswith("আমার")
    assert fix_bengali_asr("আমি আমি বলছি") == "আমি বলছি"
    # Must not glue অনেক + কিছু across the shared ক
    assert "অনেক কিছু" in fix_bengali_asr("অনেক কিছু ভালো")
