"""C9 generation with history references, structured output and human review."""
import argparse
import json
import re
import sys
from pathlib import Path

from data_generation import run_generation

# C9 semantic rules, with explicit reference IDs and a structured transport format.
SYSTEM_PROMPT = '''Generate conversational game recommendation evaluation data.
Follow these rules:
1. Generate 2-5 complete User/Assistent rounds (4, 6, 8, or 10 messages).
2. Start with User and alternate User and Assistent strictly.
3. Mention no game outside the supplied history, except the final <item> placeholder.
4. Refer to historical games only as [H1], [H2], etc., using the supplied mapping.
   Never spell out game titles in dialogue text.
5. The assistant does not know the user's history initially; the user must express
   historical preferences in the conversation.
6. Only the user initially knows the supplied target attributes. The assistant
   must infer them from the user's messages. Do not invent target attributes.
7. Reveal information gradually and keep user messages short and natural.
8. Never reveal the target title or its ID.
9. End with an Assistent recommendation containing <item>. This placeholder must
   not appear in any earlier message. Do not continue after that recommendation.
10. Each message's mentioned_ids must exactly list its historical references,
    without brackets, or be empty when there are none. <item> is not a history ID.
Return the requested JSON object with a messages array.'''
REFERENCE = re.compile(r'\[(H\d+)\]')


def conversation_schema(ids):
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {'messages': {
            'type': 'array', 'minItems': 4, 'maxItems': 10,
            'items': {
                'type': 'object', 'additionalProperties': False,
                'properties': {
                    'role': {'type': 'string', 'enum': ['User', 'Assistent']},
                    'text': {'type': 'string'},
                    'mentioned_ids': {'type': 'array', 'items': {'type': 'string', 'enum': ids}},
                },
                'required': ['role', 'text', 'mentioned_ids'],
            },
        }},
        'required': ['messages'],
    }


def prepare(row, catalog, history, rng):
    fields = ['release_date', 'tags', 'price', 'description']
    metadata = catalog.loc[row['target'], fields]
    if metadata.ndim != 1:
        raise ValueError('Ambiguous target title: multiple catalog entries')
    info = json.loads(metadata.to_json(force_ascii=False))
    id_to_title = {f'H{i + 1}': title for i, title in enumerate(history)}
    prompt = ('Historical games:\n' + '\n'.join(f'{key} = {title}' for key, title in id_to_title.items())
              + '\n\nTarget information:\n' + json.dumps(info, ensure_ascii=False))
    return prompt, {
        'target_info': info, 'id_to_title': id_to_title, 'system_prompt': SYSTEM_PROMPT,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'conversation', 'strict': True, 'schema': conversation_schema(list(id_to_title)),
        }},
    }


def context_from_messages(messages):
    """Mechanical export checks; human review determines semantic acceptability."""
    if not isinstance(messages, list) or len(messages) not in (4, 6, 8, 10):
        raise ValueError('Expected 2-5 complete dialogue rounds')
    for i, message in enumerate(messages):
        role = 'User' if i % 2 == 0 else 'Assistent'
        if not isinstance(message, dict) or message.get('role') != role:
            raise ValueError('Invalid message role sequence')
        if not isinstance(message.get('text'), str) or not message['text'].strip():
            raise ValueError('Empty or invalid message text')
        if re.search(r'\[H[^\]]*\]', message['text']):
            raise ValueError('Unresolved history reference')
    if '<item>' not in messages[-1]['text']:
        raise ValueError('Missing final target placeholder')
    if any('<item>' in m['text'] for m in messages[:-1]):
        raise ValueError('Premature target placeholder')
    return '\n'.join(f'{"User" if i % 2 == 0 else "Assistant"}: {m["text"]}'
                     for i, m in enumerate(messages[:-1]))


def parse(raw, row, history, details):
    """Restore references, flag issues for review; never assert semantic validity."""
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {'messages'} or not isinstance(obj['messages'], list):
        raise ValueError('Expected a JSON object containing messages')
    mapping = details['id_to_title']
    messages, flags = [], []
    used_any = False
    for i, msg in enumerate(obj['messages']):
        if not isinstance(msg, dict) or set(msg) != {'role', 'text', 'mentioned_ids'}:
            raise ValueError('Message must contain role, text and mentioned_ids only')
        if not isinstance(msg['text'], str) or msg['role'] not in ('User', 'Assistent'):
            raise ValueError('Invalid role or text')
        ids = msg['mentioned_ids']
        if not isinstance(ids, list) or any(not isinstance(key, str) or key not in mapping for key in ids):
            raise ValueError('Unknown or invalid mentioned_ids')
        used = REFERENCE.findall(msg['text'])
        if any(key not in mapping for key in used):
            raise ValueError('Unknown history reference in text')
        used_any |= bool(used)
        if set(used) != set(ids):
            flags.append(f'message {i + 1}: mentioned_ids/reference mismatch')
        # Check before restoration, so legitimate history titles cannot trigger target-substring rejection.
        if re.search(r'(?<!\w)' + re.escape(row['target']) + r'(?!\w)', msg['text'], re.I):
            flags.append(f'message {i + 1}: possible target-title mention; human judgment required')
        if any(title.strip() and title.strip().casefold() in msg['text'].casefold() for title in history):
            flags.append(f'message {i + 1}: historical title spelled out instead of reference')
        text = REFERENCE.sub(lambda match: mapping[match[1]], msg['text'])
        restored = {'role': msg['role'], 'text': text}
        if ids:
            restored['mentioned_items'] = [mapping[key] for key in ids]
        messages.append(restored)
    if not used_any:
        flags.append('No history reference in dialogue')
    context = None
    try:
        context = context_from_messages(messages)
    except ValueError as exc:
        flags.append(str(exc))
    return {'messages': messages, 'context': context, 'review_flags': flags}


def export_reviewed(source, output):
    """Export only explicitly approved records after every record has a decision."""
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    if not records:
        raise ValueError('Review file is empty')
    if any(not isinstance(record, dict) for record in records):
        raise ValueError('Review records must be objects')
    total = records[0].get('total_samples')
    if type(total) is not int or total != len(records) or any(r.get('total_samples') != total for r in records):
        raise ValueError('Review file must contain the full set of generated records')
    results, seen = [], set()
    for record in records:
        number = record.get('line')
        if type(number) is not int or not 1 <= number <= total:
            raise ValueError('Invalid sample line number')
        if number in seen:
            raise ValueError(f'Duplicate sample line: {number}')
        seen.add(number)
        decision = record.get('review_status')
        notes = record.get('review_notes')
        if decision not in ('approved', 'rejected') or not isinstance(notes, str) or not notes.strip():
            raise ValueError(f'Sample {number}: set approved/rejected and document review_notes')
        if decision == 'rejected':
            continue
        if record.get('generation_status') != 'complete':
            raise ValueError(f'Sample {number}: incomplete/refused API response cannot be approved')
        target = record.get('target')
        if not isinstance(target, str) or not target or target != record.get('input', {}).get('target'):
            raise ValueError(f'Sample {number}: original target must be preserved')
        results.append({'context': context_from_messages(record['messages']), 'target': target})
    if not results:
        raise ValueError('No approved samples to export')
    # Validate all decisions before creating the output; never overwrite existing experiments.
    with output.open('x', encoding='utf-8') as dest:
        for result in results:
            dest.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f'Exported {len(results)}/{len(records)} reviewed samples to {output}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'export-reviewed':
        parser = argparse.ArgumentParser(description='Export human-reviewed C9 conversations')
        parser.add_argument('review_file', type=Path)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        export_reviewed(args.review_file, args.output)
    else:
        run_generation('retrival', prepare, parse)
