"""
recursive_symmetric_split.py

A ready-to-run Python script to recursively split strings by symmetric symbols (e.g., {}, [], (), <>) including nested cases.
Usage:
    python recursive_symmetric_split.py "your string here"
    # or run and follow the prompt for input
"""
import sys
import re
from typing import List, Tuple


SYMMETRIC_PAIRS = {
    '{': '}',
    '[': ']',
    '(': ')',
    '<': '>'
}
OPENING = set(SYMMETRIC_PAIRS.keys())
CLOSING = set(SYMMETRIC_PAIRS.values())


def recursive_symmetric_split(s: str, pairs=SYMMETRIC_PAIRS) -> List[str]:
    """
    Recursively split a string by symmetric symbols, preserving nested structures.
    Returns a flat list of tokens, including the symbols themselves.
    This version correctly handles multiple and nested symmetric groups in a single string.
    If a symmetric group is not closed, the unmatched part is preserved as a single token.
    """
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] in OPENING:
            open_sym = s[i]
            close_sym = pairs[open_sym]
            start = i
            depth = 1
            i += 1
            inner_start = i
            while i < n and depth > 0:
                if s[i] == open_sym:
                    depth += 1
                elif s[i] == close_sym:
                    depth -= 1
                i += 1
            if depth == 0:
                tokens.append(open_sym)
                inner = s[inner_start:i-1]
                tokens.extend(recursive_symmetric_split(inner, pairs))
                tokens.append(close_sym)
            else:

                tokens.append(s[start:])
                break
        elif s[i] in CLOSING:
            tokens.append(s[i])
            i += 1
        else:
            start = i
            while i < n and s[i] not in OPENING and s[i] not in CLOSING:
                i += 1
            tokens.append(s[start:i])
    return [t for t in tokens if t]


def split_by_space_and_symmetric(s: str) -> List[str]:
    """
    Split the input string by spaces, then for each token, if the first character is an opening symbol and the last character is the corresponding closing symbol (according to SYMMETRIC_PAIRS),
    split out the opening symbol, the inner content, and the closing symbol as separate tokens. Otherwise, keep the token as is.
    """
    tokens = []
    for part in s.split():
        if (
            len(part) > 1
            and part[0] in SYMMETRIC_PAIRS
            and part[-1] == SYMMETRIC_PAIRS[part[0]]
        ):
            tokens.append(part[0])
            tokens.append(part[1:-1])
            tokens.append(part[-1])
        else:
            tokens.append(part)
    return [t for t in tokens if t]


def main():
    if len(sys.argv) > 1:
        s = sys.argv[1]
    else:
        s = input("Enter a string to split by symmetric symbols: ")
    tokens = split_by_space_and_symmetric(s)
    print("Split tokens:")
    print(tokens)

if __name__ == "__main__":
    main()
