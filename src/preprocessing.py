# -*- coding: utf-8 -*-
"""
===================================================================
PHASE 2: Text Preprocessing -- Spam Detection Project
===================================================================

PURPOSE OF THIS FILE:
    Machine learning models are MATH machines.
    They cannot read "FREE PRIZE WIN NOW!!" -- they only understand
    numbers.  Before converting text to numbers (Phase 3), we must
    CLEAN the text so that:

        1. The same word in different forms is treated identically
           "FREE", "Free", "free" --> all become "free"

        2. Noise is removed (URLs, phone numbers, punctuation)
           These don't help the model generalize across messages.

        3. The vocabulary size is reduced
           Fewer unique tokens = faster model, less overfitting.

PIPELINE (what this file builds):
    Raw Text
        |
        v
    convert_to_lowercase()    "FREE Entry Win!" --> "free entry win!"
        |
        v
    remove_urls()             removes http://, www. links
        |
        v
    remove_numbers()          "call 0800" --> "call "
        |
        v
    remove_punctuation()      "free!!!" --> "free"
        |
        v
    remove_stopwords()        "I will go" --> "go"
        |
        v
    tokenize()                "go win prize" --> ["go", "win", "prize"]
        |
        v
    Clean Token List  <-- ready for TF-IDF in Phase 3

FUNCTIONS IN THIS FILE:
    convert_to_lowercase(text)   -> str
    remove_urls(text)            -> str
    remove_numbers(text)         -> str
    remove_punctuation(text)     -> str
    remove_stopwords(text)       -> str
    tokenize(text)               -> list[str]
    preprocess_text(text)        -> list[str]   (the main pipeline)
===================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------
# re     --> "regular expressions" -- a powerful way to find and
#            replace patterns in text (like finding all URLs)
# string --> contains useful constants like string.punctuation
#            which is: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# nltk   --> Natural Language Toolkit -- provides stopwords list
# ---------------------------------------------------------------

import re
import string
import nltk

# Download NLTK stopwords (only needed once, safe to call every time)
# Stopwords = very common English words that carry no useful meaning
# for spam detection: "the", "a", "is", "in", "it", "to", "and" ...
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# Load the English stopwords into a Python set for fast lookup
# Using a set (not a list) because "x in set" is O(1) vs O(n) in list
STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------
# Let's peek at what stopwords look like
STOPWORDS_PREVIEW = sorted(list(STOPWORDS))[:10]


# ===================================================================
# FUNCTION 1: convert_to_lowercase
# ===================================================================
def convert_to_lowercase(text: str) -> str:
    """
    Convert all characters in text to lowercase.

    WHY DO THIS?
        The word "FREE" and "free" and "Free" are the SAME word.
        But without lowercasing, the model treats them as 3 different
        features -- that's wasteful and inaccurate.

        Example:
            "FREE entry WIN now" --> "free entry win now"

        This is always the FIRST step because later steps (like
        stopwords removal) compare against lowercase word lists.

    EDGE CASES:
        - None input  --> returns empty string (safe)
        - Empty string --> returns empty string (safe)

    Args:
        text (str): Raw input text

    Returns:
        str: Lowercased text
    """
    if not text or not isinstance(text, str):
        return ""
    return text.lower()


# ===================================================================
# FUNCTION 2: remove_urls
# ===================================================================
def remove_urls(text: str) -> str:
    """
    Remove all URLs from the text (http://, https://, www.).

    WHY DO THIS?
        Spam messages often contain unique URLs to phishing sites:
            "Visit http://free-prize-win-now.xyz/claim12345"

        These URLs are:
          (a) Too specific -- a URL in one spam won't appear in
              another spam message with a different ID or domain.
          (b) Not generalizable -- the model can't learn "this URL
              pattern = spam" because every spam has a unique URL.
          (c) Noisy -- they introduce rare tokens that hurt accuracy.

        What matters is NOT the URL itself, but the PRESENCE of a URL.
        (We could add a feature "has_url=1" in Phase 3 as a numeric
        feature -- but for now we just remove them.)

    HOW IT WORKS (regex breakdown):
        r'http\S+' --> match "http" followed by any non-space chars
        r'www\.\S+' --> match "www." followed by any non-space chars
        re.sub(pattern, replacement, text) --> replace match with " "

    Args:
        text (str): Text possibly containing URLs

    Returns:
        str: Text with URLs removed
    """
    if not text or not isinstance(text, str):
        return ""
    # Remove http:// and https:// URLs
    text = re.sub(r"http\S+", " ", text)
    # Remove www. URLs (in case http:// is missing)
    text = re.sub(r"www\.\S+", " ", text)
    return text


# ===================================================================
# FUNCTION 3: remove_numbers
# ===================================================================
def remove_numbers(text: str) -> str:
    """
    Remove all numeric digits from the text.

    WHY DO THIS?
        Spam messages often contain specific numbers:
            "Call 0800 123 456"  or  "Win £1000"  or  "50% OFF"

        These specific numbers:
          (a) Are unique to each spam -- "0800123456" won't appear
              in a new spam with a different phone number.
          (b) Create many unique tokens that the model sees rarely,
              making it hard to generalize.
          (c) The WORD "call" or "win" carries the signal -- not
              the specific number following it.

        Note: Some practitioners KEEP numbers or replace them with
        a placeholder like "NUM". For now we remove them.

    HOW IT WORKS (regex):
        r'\d+' --> match one or more digits (0-9)
        Replace with " " (space) so words don't accidentally merge.

    Args:
        text (str): Text possibly containing numbers

    Returns:
        str: Text with all digits removed
    """
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\d+", " ", text)


# ===================================================================
# FUNCTION 4: remove_punctuation
# ===================================================================
def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation characters from the text.

    WHY DO THIS?
        Spam messages are punctuation-heavy:
            "FREE!!! WIN NOW!!! Call IMMEDIATELY!!!"

        Punctuation like "!", "£", "#" adds noise:
          (a) "free!!!" and "free" should be the same token
          (b) Punctuation attached to words creates fake unique tokens
              e.g., "win!" vs "win" vs "win," = 3 tokens for 1 word
          (c) Rare punctuation characters create noise

        We already removed URLs and numbers -- punctuation like ":"
        and "/" in URLs won't accidentally create "http" becoming
        "http:" with the colon attached.

    HOW IT WORKS:
        string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        str.maketrans()     = builds a character translation table
        text.translate()    = applies the table (very fast!)

        Alternative: re.sub(r'[^\w\s]', '', text)
        Both work -- maketrans is slightly faster for full text.

    Args:
        text (str): Text possibly containing punctuation

    Returns:
        str: Text with punctuation replaced by spaces
    """
    if not text or not isinstance(text, str):
        return ""
    # Build a translation table: every punctuation char -> " " (space)
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


# ===================================================================
# FUNCTION 5: remove_stopwords
# ===================================================================
def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords from the text.

    WHY DO THIS?
        Stopwords are very frequent words that appear in BOTH spam
        and ham messages equally -- they carry NO useful signal:

            "I am going to the store"  (ham)
            "You are the lucky winner" (spam)

        Words like "I", "am", "going", "to", "the", "you", "are"
        appear in both. They don't help the model distinguish.
        Removing them:
          (a) Reduces vocabulary size (faster model)
          (b) Forces the model to focus on meaningful words
          (c) Reduces noise in TF-IDF weights

        IMPORTANT: We remove stopwords AFTER punctuation removal.
        If we did it before, "don't" might not match "dont"
        after punctuation removal splits it.

    STOPWORDS EXAMPLES (from NLTK English):
        "a", "an", "the", "is", "are", "was", "were",
        "i", "me", "my", "we", "you", "he", "she", "it",
        "to", "of", "in", "for", "on", "with", "at" ...
        (Total: ~179 words)

    HOW IT WORKS:
        1. Split text into words
        2. Keep only words NOT in the stopwords set
        3. Rejoin into a string

    Args:
        text (str): Space-separated text

    Returns:
        str: Text with stopwords removed
    """
    if not text or not isinstance(text, str):
        return ""
    words = text.split()
    # Keep word only if it's NOT in the stopwords set
    # and if it has at least 2 characters (removes stray single letters)
    filtered = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return " ".join(filtered)


# ===================================================================
# FUNCTION 6: tokenize
# ===================================================================
def tokenize(text: str) -> list:
    """
    Split the cleaned text into a list of individual words (tokens).

    WHY DO THIS?
        ML models work with features. For text classification,
        each WORD is a feature.

        "free prize win" --> ["free", "prize", "win"]

        This list of tokens is what TF-IDF (Phase 3) processes:
        - Each unique word becomes a column in a matrix
        - The value = how important that word is in this message

        WHY NOT JUST KEEP THE STRING?
            TF-IDF's CountVectorizer can tokenize internally, but
            having our own tokenizer gives us full control and lets
            us apply custom cleaning before vectorization.

    WHAT IS A TOKEN?
        A token is a single unit of meaning -- usually a word.
        "free prize win" has 3 tokens: ["free", "prize", "win"]

    Args:
        text (str): Clean, preprocessed text string

    Returns:
        list[str]: List of word tokens (empty list if input is empty)
    """
    if not text or not isinstance(text, str):
        return []
    # Split on any whitespace (handles multiple spaces, tabs, newlines)
    tokens = text.split()
    # Final filter: only keep non-empty strings
    return [t for t in tokens if t]


# ===================================================================
# MAIN PIPELINE: preprocess_text
# ===================================================================
def preprocess_text(text: str) -> list:
    """
    THE MAIN PREPROCESSING FUNCTION -- runs the full pipeline.

    This chains all the individual steps together in the correct order.
    This is what you call on every message before training or predicting.

    PIPELINE ORDER (ORDER MATTERS!):

        Raw Text
          |
          v  Step 1: Lowercase first -- so later steps work correctly
        convert_to_lowercase()
          |
          v  Step 2: Remove URLs -- do before punctuation (URLs have . : / )
        remove_urls()
          |
          v  Step 3: Remove numbers -- do before punctuation (numbers alone)
        remove_numbers()
          |
          v  Step 4: Remove punctuation -- clean leftover symbols
        remove_punctuation()
          |
          v  Step 5: Remove stopwords -- after cleaning, words are pure
        remove_stopwords()
          |
          v  Step 6: Tokenize -- split into list for TF-IDF
        tokenize()
          |
          v
        [list of clean tokens]

    WHY THIS SPECIFIC ORDER?
        - Lowercase FIRST ensures stopwords match ("The" vs "the")
        - URLs before punctuation prevents "http:" splitting wrong
        - Numbers before punctuation prevents "123." leaving "."
        - Punctuation before stopwords ensures "don't" -> "dont"
          -> after split -> "dont" (which isn't a stopword, which is fine)
        - Stopwords last (after cleaning) removes genuinely useless words
        - Tokenize always last -- everything else works on strings

    Args:
        text (str): Raw SMS text (could be messy, noisy, any case)

    Returns:
        list[str]: Clean list of meaningful tokens, ready for TF-IDF

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


# ===================================================================
# HELPER: tokens_to_string
# ===================================================================
def tokens_to_string(tokens: list) -> str:
    """
    Convert a token list back to a single string.

    WHY NEEDED?
        Some vectorizers (like TF-IDF's TfidfVectorizer) expect a
        string input, not a list.  Rather than re-joining everywhere,
        we provide this small helper.

        Usage:
            tokens = preprocess_text(msg)
            clean_str = tokens_to_string(tokens)
            # pass clean_str to TfidfVectorizer

    Args:
        tokens (list[str]): List of word tokens

    Returns:
        str: Space-joined string of tokens
    """
    if not tokens:
        return ""
    return " ".join(tokens)


def preprocess_to_string(text: str) -> str:
    """
    Full pipeline that returns a CLEAN STRING (not a list).

    This combines preprocess_text() + tokens_to_string().
    TfidfVectorizer in Phase 3 expects strings, so we'll use this
    when building our feature matrix.

    Args:
        text (str): Raw SMS text

    Returns:
        str: Clean, space-joined string of meaningful words
    """
    return tokens_to_string(preprocess_text(text))


# ===================================================================
# DEMONSTRATION -- runs when you execute this file directly
# ===================================================================
if __name__ == "__main__":

    SEP  = "=" * 68
    SEP2 = "-" * 68

    print(SEP)
    print("  PHASE 2: TEXT PREPROCESSING -- STEP-BY-STEP DEMO")
    print(SEP)

    # ---------------------------------------------------------------
    # Show what stopwords look like
    # ---------------------------------------------------------------
    print(f"""
  WHAT ARE STOPWORDS?
    NLTK gives us {len(STOPWORDS)} common English words to remove.
    First 15: {sorted(list(STOPWORDS))[:15]}
    ...and {len(STOPWORDS) - 15} more.
    These words appear in BOTH spam and ham equally,
    so they carry NO useful signal for classification.
""")

    # ---------------------------------------------------------------
    # Demo message: a realistic spam SMS
    # ---------------------------------------------------------------
    raw_message = (
        "FREE ENTRY in 2 a weekly competition to win FA Cup Final tkts "
        "21st May 2005. Text FA to 87121 to receive entry question(std txt rate) "
        "T&C's apply 08452810075over18's. Visit http://win.example.com"
    )

    print(SEP)
    print("  DEMO: TRACING ONE MESSAGE THROUGH THE FULL PIPELINE")
    print(SEP)
    print(f"\n  ORIGINAL MESSAGE:\n  {raw_message}\n")

    # ---------------------------------------------------------------
    # Show each step's transformation
    # ---------------------------------------------------------------
    steps = []

    step1 = convert_to_lowercase(raw_message)
    steps.append(("Step 1 | convert_to_lowercase()", step1))

    step2 = remove_urls(step1)
    steps.append(("Step 2 | remove_urls()", step2))

    step3 = remove_numbers(step2)
    steps.append(("Step 3 | remove_numbers()", step3))

    step4 = remove_punctuation(step3)
    steps.append(("Step 4 | remove_punctuation()", step4))

    step5 = remove_stopwords(step4)
    steps.append(("Step 5 | remove_stopwords()", step5))

    step6 = tokenize(step5)
    steps.append(("Step 6 | tokenize()", str(step6)))

    for label, result in steps:
        print(f"  [{label}]")
        print(f"  --> {result}")
        print()

    # ---------------------------------------------------------------
    # What changed?
    # ---------------------------------------------------------------
    original_words = raw_message.split()
    final_tokens   = preprocess_text(raw_message)
    removed_count  = len(original_words) - len(final_tokens)

    print(SEP2)
    print(f"  WHAT CHANGED:")
    print(f"    Original word count  : {len(original_words)}")
    print(f"    Final token count    : {len(final_tokens)}")
    print(f"    Words removed        : {removed_count} ({removed_count/len(original_words)*100:.0f}%)")
    print(f"    Meaningful tokens    : {final_tokens}")
    print(f"    As string            : '{tokens_to_string(final_tokens)}'")

    # ---------------------------------------------------------------
    # Test with multiple messages
    # ---------------------------------------------------------------
    test_messages = [
        ("SPAM", "Congratulations! You've WON a £1,000 Tesco gift card. "
                 "Go to http://promo.tesco-winner.uk NOW!"),
        ("SPAM", "URGENT! Your Mobile number has been awarded a £2,000 Bonus Caller Prize. "
                 "CALL 09061743806 from land line. Claim code KL341."),
        ("HAM",  "Hey, are you coming to the party tonight? Let me know!"),
        ("HAM",  "I'll be late. Can you please wait for me at the entrance?"),
        ("HAM",  "Ok sounds good. I'll call you when I'm on my way."),
    ]

    print(f"\n\n{SEP}")
    print("  APPLYING PIPELINE TO 5 REAL MESSAGES")
    print(SEP)

    for i, (label, msg) in enumerate(test_messages, 1):
        tokens = preprocess_text(msg)
        clean  = tokens_to_string(tokens)
        print(f"\n  [{i}] [{label}]")
        print(f"  BEFORE : {msg[:80]}{'...' if len(msg) > 80 else ''}")
        print(f"  AFTER  : {clean}")
        print(f"  TOKENS : {tokens}")

    # ---------------------------------------------------------------
    # Edge case handling
    # ---------------------------------------------------------------
    print(f"\n\n{SEP}")
    print("  EDGE CASE TESTS")
    print(SEP2)

    edge_cases = [
        ("Empty string",  ""),
        ("None input",    None),
        ("Only numbers",  "123 456 789"),
        ("Only punctuation", "!!! ??? ### +++"),
        ("Only stopwords",   "the is and a of to"),
        ("Single keyword",   "FREE"),
    ]

    for label, case in edge_cases:
        result = preprocess_text(case)
        print(f"  {label:20s}  -->  {result}")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print(f"\n\n{SEP}")
    print("  PHASE 2 SUMMARY")
    print(SEP)
    print(f"""
  FUNCTIONS BUILT:
    convert_to_lowercase()  "FREE Win" --> "free win"
    remove_urls()           "go to http://x.com" --> "go to"
    remove_numbers()        "call 0800" --> "call"
    remove_punctuation()    "win!!!" --> "win"
    remove_stopwords()      "I will win" --> "win"
    tokenize()              "free win" --> ["free", "win"]

    preprocess_text()       Master pipeline: raw text --> token list
    preprocess_to_string()  Master pipeline: raw text --> clean string

  WHY THE ORDER MATTERS:
    Lowercase first  --> ensures stopwords like "The" match "the"
    URLs second      --> remove before punctuation splits the URL
    Numbers third    --> digits removed before punctuation cleanup
    Punctuation 4th  --> clears leftover symbols
    Stopwords 5th    --> now working with clean, lowercase words
    Tokenize last    --> after all cleanup, split into list

  WHAT'S NEXT -- PHASE 3: VECTORIZATION (TF-IDF)

    Problem: ML models need NUMBERS, not word lists.
    Solution: TF-IDF (Term Frequency - Inverse Document Frequency)

    Each cleaned message becomes a ROW in a numeric matrix.
    Each unique word becomes a COLUMN.
    The value at [row, column] = TF-IDF score of that word in that message.

    Example (tiny):
                     "free"  "win"  "prize"  "call"  "tonight"
    spam message 1:  [ 0.8,   0.6,   0.0,    0.4,    0.0     ]
    spam message 2:  [ 0.0,   0.7,   0.9,    0.0,    0.0     ]
    ham  message 1:  [ 0.0,   0.0,   0.0,    0.0,    0.5     ]

    The model LEARNS that high values in "free", "win", "prize"
    columns predict spam!
""")
    print(SEP)
    print("  Phase 2 Complete!  Ready for Phase 3: Vectorization (TF-IDF)")
    print(SEP)
