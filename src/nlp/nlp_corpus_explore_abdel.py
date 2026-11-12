"""nlp_corpus_explore_abdel.py - Module 3 Custom Project.

Author: Abdelhafidh Mahouel
Date: 2026-11

Purpose

  Apply exploratory text analysis to manufacturing continuous improvement.
  The corpus is a set of short shop-floor reports, the kind of free-text
  notes written in quality alerts, maintenance work orders, safety
  observations, and lean / kaizen event summaries.

  The goal is to see whether simple text techniques (token frequency,
  category comparison, co-occurrence, and bigrams) can surface the
  vocabulary that separates each improvement area and point an engineer
  toward recurring problems, before any model is used.

Analytical Questions

- What words dominate each continuous improvement category?
- How does the vocabulary of quality, maintenance, safety, and lean differ?
- What words appear near shared shop-floor terms like "line" and "operator"?
- Which two-word phrases (bigrams) repeat, and what do they suggest?
- What cannot be concluded from frequency counts alone?

Notes:
- This module focuses on exploratory analysis (EDA), not modeling.
- The corpus is fictional but written to reflect realistic plant reports.

Data: data/ci_reports_abdel.csv (doc_id, category, text)

Run from root project folder with:

  uv run python -m nlp.nlp_corpus_explore_abdel
"""

# ============================================================
# Section 1. Setup and Imports
# ============================================================

from collections import defaultdict
import logging
from pathlib import Path

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
import polars as pl

print("Imports complete.")


# ============================================================
# Configure Logging
# ============================================================

LOG: logging.Logger = get_logger("CI", level="DEBUG")

ROOT_PATH: Path = Path.cwd()
SCRIPTS_PATH: Path = ROOT_PATH / "scripts"
IMAGES_PATH: Path = ROOT_PATH / "docs" / "images"

log_header(LOG, "MODULE 3: MANUFACTURING CI CORPUS EXPLORATION (ABDEL)")
LOG.info("START script.....")

log_path(LOG, "ROOT_PATH", ROOT_PATH)
log_path(LOG, "IMAGES_PATH", IMAGES_PATH)

LOG.info("Logger configured.")

# ============================================================
# Section 2. Load Corpus (Labeled Shop-Floor Reports)
# ============================================================

# CUSTOM: The corpus lives in its own CSV file, separate from the code.
# Four categories, six documents each (24 documents total).
# Some words (line, operator, station) are shared across categories
# on purpose, so co-occurrence can show how the same word is used differently.
# The path is built from this file's location so the script and the tests
# find the data no matter which folder they run from.

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = PROJECT_ROOT / "data" / "ci_reports_abdel.csv"
log_path(LOG, "DATA_PATH", DATA_PATH)

corpus_df: pl.DataFrame = pl.read_csv(DATA_PATH)
corpus: list[dict[str, str]] = corpus_df.select(["category", "text"]).to_dicts()


CATEGORIES: list[str] = ["quality", "maintenance", "safety", "lean"]

# Show results
print(f"Corpus contains {len(corpus)} documents.")
for category in CATEGORIES:
    count = sum(1 for doc in corpus if doc["category"] == category)
    print(f"  {category}: {count} documents")


# ============================================================
# Section 3. Tokenize and Clean Text
# ============================================================

# Tokenization splits text into word-like units.

# CUSTOM: Stop words are very common words that carry little meaning
# ("the", "and", "was"). In the example corpus "the" was the top token,
# which says nothing about the categories. Removing stop words lets
# the process vocabulary (defect, downtime, waste) rise to the top.
STOP_WORDS: set[str] = {
    "the", "and", "was", "were", "for", "from", "with", "are", "not",
    "out", "two", "four", "after", "before", "during", "where", "between",
    "across", "closer", "about", "into", "this", "that", "has", "had",
    "found",  # reporting verb used in every category (audit found, inspection found)
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Tokenize the input text into a list of cleaned tokens.

    Lowercase, split on whitespace, strip punctuation,
    and drop very short tokens (length <= 2).
    """
    tokens = text.lower().split()
    cleaned = [t.strip(".,:;!?()[]\"'") for t in tokens]
    return [t for t in cleaned if len(t) > 2]


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Remove stop words from a list of tokens."""
    return [t for t in tokens if t not in STOP_WORDS]


def prepare(text: str) -> list[str]:
    """Tokenize text and remove stop words (the full cleaning step)."""
    return remove_stop_words(tokenize(text))


# Build one record per token with its category.
records_list: list[dict[str, str]] = []

for doc in corpus:
    for token in prepare(doc["text"]):
        records_list.append({"category": doc["category"], "token": token})

token_df: pl.DataFrame = pl.DataFrame(records_list)

# Show results
print("Tokenization complete.")
print(f"Total tokens after cleaning: {token_df.height}")
print(token_df.head(10))


# ============================================================
# Section 4. Compute Global Token Frequencies
# ============================================================

# Frequency distribution = how often each token appears.

global_freq_df: pl.DataFrame = (
    token_df.group_by("token").len().sort(["len", "token"], descending=[True, False])
)

# Show results
print("Top global tokens:")
print(global_freq_df.head(10))


# ============================================================
# Section 5. Compute Token Frequencies by Category
# ============================================================

# Compare token usage across categories.

category_freq_df: pl.DataFrame = (
    token_df.group_by(["category", "token"])
    .len()
    .sort(["category", "len", "token"], descending=[False, True, False])
)

# CUSTOM: Find which categories each token appears in.
# A token used in only one category is a "signature" word for it.
token_spread_df: pl.DataFrame = (
    token_df.group_by("token")
    .agg(
        pl.col("category").n_unique().alias("num_categories"),
        pl.len().alias("total"),
    )
    .sort(["num_categories", "total", "token"], descending=[True, True, False])
)

# Show results
print("Top tokens by category:")
print(category_freq_df.head(12))

print("\nTokens shared across the most categories:")
print(token_spread_df.filter(pl.col("num_categories") > 1))


# ============================================================
# Section 6. Identify Top Tokens per Category
# ============================================================

top_per_category_dict: dict[str, list[str]] = {}

for category in CATEGORIES:
    subset_df = category_freq_df.filter(pl.col("category") == category).head(5)
    top_tokens_list = subset_df["token"].to_list()
    top_per_category_dict[category] = top_tokens_list

    print(f"{category.upper()} top tokens: {top_tokens_list}")


# ============================================================
# Section 7. Analyze Co-occurrence (Context Windows)
# ============================================================

# MODIFIED (Phase 4): co-occurrence is a reusable function
# so window sizes can be compared side by side.
# WINDOW_SIZE controls how many tokens on each side of a target token
# count as its context. The target token itself is not included.

WINDOW_SIZE: int = 2
WINDOW_SIZES_TO_COMPARE: list[int] = [2, 3]

# CUSTOM: Shared shop-floor words plus key words for each category.
TARGET_TOKENS: list[str] = ["line", "operator", "defect", "downtime", "waste"]


def build_co_occurrence(
    documents: list[dict[str, str]], window_size: int
) -> dict[str, list[str]]:
    """Map each token to the tokens within window_size positions of it."""
    result: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        tokens = prepare(doc["text"])
        for i, token in enumerate(tokens):
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for ctx in tokens[start:end]:
                if ctx != token:
                    result[token].append(ctx)
    return result


co_occurrence_dict: dict[str, list[str]] = build_co_occurrence(corpus, WINDOW_SIZE)

co_occurrence_by_window: dict[int, dict[str, list[str]]] = {
    size: build_co_occurrence(corpus, size) for size in WINDOW_SIZES_TO_COMPARE
}

print("\nCO-OCCURRENCE COMPARISON BY WINDOW SIZE:")
for target in TARGET_TOKENS:
    print(f"\nTarget: '{target}'")
    previous: set[str] = set()
    for size in WINDOW_SIZES_TO_COMPARE:
        context_list = co_occurrence_by_window[size][target]
        unique_context = sorted(set(context_list))
        added = sorted(set(unique_context) - previous) if previous else []
        print(
            f"  window={size}: {len(context_list)} context tokens, "
            f"{len(unique_context)} unique -> {unique_context}"
        )
        if added:
            print(f"    new at window={size}: {added}")
        previous = set(unique_context)


# ============================================================
# Section 8. Create Bigrams (Local Word Pairs) and Compute Frequencies
# ============================================================

# Bigrams capture pairs of consecutive tokens.
# CUSTOM: Bigrams are tracked by category, because manufacturing
# vocabulary is full of two-word terms (root cause, lockout tagout,
# preventive maintenance, changeover time).

bigram_records: list[dict[str, str]] = []
bigrams_list: list[tuple[str, str]] = []

for doc in corpus:
    tokens = prepare(doc["text"])
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        bigrams_list.append(pair)
        bigram_records.append(
            {"category": doc["category"], "bigram": f"{pair[0]} {pair[1]}"}
        )

bigram_df: pl.DataFrame = pl.DataFrame(bigram_records)

bigram_freq_df: pl.DataFrame = (
    bigram_df.group_by("bigram").len().sort(["len", "bigram"], descending=[True, False])
)

# Show results
print("\nTop bigrams:")
print(bigram_freq_df.head(10))

print("\nRepeated bigrams by category:")
print(
    bigram_df.group_by(["category", "bigram"])
    .len()
    .filter(pl.col("len") > 1)
    .sort(["category", "len"], descending=[False, True])
)


# ============================================================
# Section 9. Visualize Token Frequencies
# ============================================================

# CUSTOM: One panel per category so vocabulary can be compared at a glance.
# The chart is also saved to docs/images for the project report.

print("IMPORTANT: Close chart window to continue execution.")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
colors = ["#2E75B6", "#C55A11", "#BF0000", "#548235"]

for ax, category, color in zip(axes.flat, CATEGORIES, colors, strict=True):
    subset_df = category_freq_df.filter(pl.col("category") == category).head(5)
    ax.barh(subset_df["token"].to_list()[::-1], subset_df["len"].to_list()[::-1],
            color=color)  # fmt: skip
    ax.set_title(category.capitalize())
    ax.set_xlabel("Frequency")
    ax.xaxis.get_major_locator().set_params(integer=True)

fig.suptitle(
    "Mahouel: Top Tokens by Continuous Improvement Category\n"
    "(Close this window to continue)"
)
fig.tight_layout()

IMAGES_PATH.mkdir(parents=True, exist_ok=True)
chart_path = IMAGES_PATH / "ci_top_tokens_by_category_abdel.png"
fig.savefig(chart_path, dpi=120)
LOG.info(f"Saved chart to {chart_path}")

plt.show()


# ============================================================
# Section 10. Interpret Results and Identify Patterns
# ============================================================

print("\nCASE GENERAL OBSERVATIONS:")

print("- Tokens cluster by category.")
print("- Words that appear in similar contexts behave similarly.")
print("- Co-occurrence reveals contextual relationships between words.")
print("- Bigrams capture local structure beyond single tokens.")
print("- Patterns emerge before any machine learning is applied.")

print("\nMAHOUEL SPECIFIC OBSERVATIONS:")
print("- Each category has signature words: defect and inspection (quality),")
print("  downtime and preventive maintenance (maintenance), operator and")
print("  hazard words (safety), and waste and time (lean).")
print("- Removing stop words was essential: without it 'the' dominates")
print("  and hides the process vocabulary.")
print("- 'line' and 'operator' appear in several categories, but their")
print("  neighbors change: next to 'downtime' in maintenance, next to")
print("  'near miss' in safety, and next to 'cycle time' in lean.")
print("- Bigrams recover real engineering terms (preventive maintenance,")
print("  hydraulic press, root cause) that single tokens split apart.")
print("- Removing stop words can create false bigrams: 'parts line' comes")
print("  from 'parts from line' and 'parts closer to the line'.")
print("- Limitation: counts show what is mentioned, not severity, cost,")
print("  or root cause. A word appearing often does not mean it is the")
print("  biggest problem, and a small corpus can be skewed by wording.")


# ============================================================
# END
# ============================================================

LOG.info("========================")
LOG.info("Pipeline executed successfully!")
LOG.info("========================")
LOG.info("END main()")
