import re


def to_camel_case(name: str) -> str:
    """Convert an arbitrary field name into camelCase.

    Splits on any non-alphanumeric separators, lowercases the first word, and
    title-cases the rest. Examples::

        "Leetcode Topic" -> "leetcodeTopic"
        "Personal Win"   -> "personalWin"
        "Category "      -> "category"
        "Date"           -> "date"
    """
    words = [word for word in re.split(r"[^0-9A-Za-z]+", name.strip()) if word]
    if not words:
        return ""

    first = words[0].lower()
    rest = [word[:1].upper() + word[1:].lower() for word in words[1:]]
    return first + "".join(rest)
