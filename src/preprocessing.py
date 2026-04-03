# -*- coding: utf-8 -*-
"""
Phase 2: Text Preprocessing
Cleans raw SMS text through a 6-step pipeline before TF-IDF vectorization.
Steps: lowercase → remove URLs → remove numbers → remove punctuation → remove stopwords → tokenize
"""

import sys
import io
import re
import string
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
STOPWORDS_PREVIEW = sorted(list(STOPWORDS))[:10]


def convert_to_lowercase(text: str) -> str:
    """Convert text to lowercase for consistent token matching."""
    if not text or not isinstance(text, str):
        return ""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove http:// and www. URLs from text."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    return text


def remove_numbers(text: str) -> str:
    """Remove all digit sequences from text."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\d+", " ", text)


def remove_punctuation(text: str) -> str:
    """Replace all punctuation characters with spaces."""
    if not text or not isinstance(text, str):
        return ""
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords and single-character tokens."""
    if not text or not isinstance(text, str):
        return ""
    words = text.split()
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return " ".join(filtered)


def tokenize(text: str) -> list:
    """Split cleaned text into a list of word tokens."""
    if not text or not isinstance(text, str):
        return []
    tokens = text.split()
    return [t for t in tokens if t]


def preprocess_text(text: str) -> list:
    """
    Run the full preprocessing pipeline on raw SMS text.

    Returns a list of clean tokens ready for TF-IDF vectorization.

    Example:
        >>> preprocess_text("FREE ENTRY! Win £1000 at http://win.com NOW!")
        ['free', 'entry', 'win']
    """
    if not text or not isinstance(text, str):
        return []

    text = convert_to_lowercase(text)   # Step 1
    text = remove_urls(text)            # Step 2
    text = remove_numbers(text)         # Step 3
    text = remove_punctuation(text)     # Step 4
    text = remove_stopwords(text)       # Step 5
    tokens = tokenize(text)             # Step 6

    return tokens


def tokens_to_string(tokens: list) -> str:
    """Join a token list into a space-separated string."""
    if not tokens:
        return ""
    return " ".join(tokens)


def preprocess_to_string(text: str) -> str:
    """Run the full pipeline and return a clean string (not a list)."""
    return tokens_to_string(preprocess_text(text))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    SEP  = "=" * 68
    SEP2 = "-" * 68

    print(SEP)
    print("  PHASE 2: TEXT PREPROCESSING -- STEP-BY-STEP DEMO")
    print(SEP)

    print(f"\n  Stopwords loaded: {len(STOPWORDS)} words")
    print(f"  Sample: {sorted(list(STOPWORDS))[:15]}\n")

    raw_message = (
        "FREE ENTRY in 2 a weekly competition to win FA Cup Final tkts "
        "21st May 2005. Text FA to 87121 to receive entry question(std txt rate) "
        "T&C's apply 08452810075over18's. Visit http://win.example.com"
    )

    print(SEP)
    print("  DEMO: TRACING ONE MESSAGE THROUGH THE PIPELINE")
    print(SEP)
    print(f"\n  ORIGINAL: {raw_message}\n")

    steps = []
    step1 = convert_to_lowercase(raw_message);  steps.append(("1 | convert_to_lowercase()", step1))
    step2 = remove_urls(step1);                 steps.append(("2 | remove_urls()", step2))
    step3 = remove_numbers(step2);              steps.append(("3 | remove_numbers()", step3))
    step4 = remove_punctuation(step3);          steps.append(("4 | remove_punctuation()", step4))
    step5 = remove_stopwords(step4);            steps.append(("5 | remove_stopwords()", step5))
    step6 = tokenize(step5);                    steps.append(("6 | tokenize()", str(step6)))

    for label, result in steps:
        print(f"  [{label}]")
        print(f"  --> {result}\n")

    original_words = raw_message.split()
    final_tokens   = preprocess_text(raw_message)
    removed_count  = len(original_words) - len(final_tokens)

    print(SEP2)
    print(f"  Original words : {len(original_words)}")
    print(f"  Final tokens   : {len(final_tokens)}  (removed {removed_count}, {removed_count/len(original_words)*100:.0f}%)")
    print(f"  Tokens         : {final_tokens}")
    print(f"  As string      : '{tokens_to_string(final_tokens)}'")

    test_messages = [
        ("SPAM", "Congratulations! You've WON a £1,000 Tesco gift card. Go to http://promo.tesco-winner.uk NOW!"),
        ("SPAM", "URGENT! Your Mobile number has been awarded a £2,000 Bonus Caller Prize. CALL 09061743806 from land line."),
        ("HAM",  "Hey, are you coming to the party tonight? Let me know!"),
        ("HAM",  "I'll be late. Can you please wait for me at the entrance?"),
        ("HAM",  "Ok sounds good. I'll call you when I'm on my way."),
    ]

    print(f"\n{SEP}")
    print("  PIPELINE APPLIED TO 5 MESSAGES")
    print(SEP)

    for i, (label, msg) in enumerate(test_messages, 1):
        tokens = preprocess_text(msg)
        clean  = tokens_to_string(tokens)
        print(f"\n  [{i}] [{label}]")
        print(f"  BEFORE : {msg[:80]}{'...' if len(msg) > 80 else ''}")
        print(f"  AFTER  : {clean}")

    print(f"\n{SEP}")
    print("  EDGE CASE TESTS")
    print(SEP2)

    edge_cases = [
        ("Empty string",       ""),
        ("None input",         None),
        ("Only numbers",       "123 456 789"),
        ("Only punctuation",   "!!! ??? ### +++"),
        ("Only stopwords",     "the is and a of to"),
        ("Single keyword",     "FREE"),
    ]

    for label, case in edge_cases:
        result = preprocess_text(case)
        print(f"  {label:20s}  -->  {result}")

    print(f"\n{SEP}")
    print("  Phase 2 Complete!  Ready for Phase 3: Vectorization (TF-IDF)")
    print(SEP)
