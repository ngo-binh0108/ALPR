"""Character set shared by training and the C++ app.

Index 0 is the CTC blank.  The exported ``alpr_models.json`` carries this exact
list, so the C++ decoder never hard-codes characters.
"""

PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
SPECIALS = "使领警学港澳挂民航应急"
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # Chinese plates never use I or O
DIGITS = "0123456789"

BLANK = "-"
CHARS = [BLANK] + list(PROVINCES + SPECIALS + LETTERS + DIGITS)
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

# plate colour head
COLORS = ["blue", "green", "yellow", "white", "black"]
COLOR2IDX = {c: i for i, c in enumerate(COLORS)}

# Valid plate formats (GA36-2018). Exported into alpr_models.json as "plate_regex": the C++ app uses it to
# drop implausible reads, so a model for another country/style only needs its own pattern here.
_P = f"[{PROVINCES}]"
_A = "[A-HJ-NP-Z]"          # letter, no I/O
_X = "[A-HJ-NP-Z0-9]"       # letter or digit
PLATE_REGEX = ("^(?:"
               f"{_P}{_A}{_X}{{5}}"                 # standard 7 (blue / yellow / black)
               f"|{_P}{_A}{_X}{{6}}"                # new-energy 8 (green)
               f"|{_P}{_A}{_X}{{4}}[学警挂港澳领]"   # suffix plates
               f"|{_P}{_A}{_X}{{4}}应急"             # emergency
               "|使[0-9]{6}|民航[A-HJ-NP-Z0-9]{5}"   # embassy / civil aviation
               ")$")


def encode(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text]


def is_valid(text: str) -> bool:
    return 6 <= len(text) <= 8 and all(c in CHAR2IDX and c != BLANK for c in text)


def ctc_greedy_decode(probs, chars=None):
    """probs: (T, C) numpy array of per-step class probabilities.

    Returns (text, confidence) where confidence is the mean max-probability of
    the emitted characters (identical to the C++ implementation).
    """
    chars = chars or CHARS
    best = probs.argmax(axis=1)
    text, confs, prev = [], [], 0
    for t, k in enumerate(best):
        k = int(k)
        if k != 0 and k != prev:
            text.append(chars[k])
            confs.append(float(probs[t, k]))
        prev = k
    conf = sum(confs) / len(confs) if confs else 0.0
    return "".join(text), conf
