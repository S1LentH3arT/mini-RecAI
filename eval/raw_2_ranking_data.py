"""Figure C10 ranking data generation with fixed 19 negatives plus target."""
import json

from data_generation import neg_item_sample, run_generation

# Faithful paraphrase of Figure C10: https://arxiv.org/html/2308.16505
# Its history-only rule is interpreted as excluding titles outside history/candidates.
PROMPT = '''Act as a user asking a recommendation assistant to rank games.
Previously played games: {history}
The user has the following {n} candidate games: {candidates}
Write a short user question requesting a personalized ranking of those candidates.
The assistant does not already know the user's history, so include that history
in the question. Include every candidate, preserving its exact title, and do not
introduce game titles outside the history and candidate list.
For example, describe games enjoyed before, present the possible next games, and
ask for their ordering according to the user's preferences.
Return only the user's question, with no answer or suggested ranking.'''


def prepare(row, catalog, history, rng):
    candidates = neg_item_sample(catalog.index, history, row['target'], 19, rng)
    return PROMPT.format(history=json.dumps(history, ensure_ascii=False), n=len(candidates),
                         candidates=json.dumps(candidates, ensure_ascii=False)), {'candidates': candidates}


def parse(raw, row, history, details):
    question = raw.strip()
    if not question or question.startswith(('```', '{', '[')):
        raise ValueError('Expected a plain-text user question')
    if any(title not in question for title in details['candidates']):
        raise ValueError('Question is missing or has altered candidate titles')
    if any(title not in question for title in history):
        raise ValueError('Question is missing or has altered history titles')
    if not any(word in question.lower() for word in ('rank', 'order', 'sort')):
        raise ValueError('Question does not request ranking')
    return question


if __name__ == '__main__':
    run_generation('ranking', prepare, parse)
