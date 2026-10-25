"""Shared sampling and generation I/O; no Agent or recommender-model imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
from pathlib import Path

import pandas as pd
from openai import APIStatusError, OpenAI, OpenAIError


def neg_item_sample(catalog_titles, history_titles, target, n, rng):
    """抽取 n 个不同负例，加入正例后打乱。"""
    titles = sorted(set(catalog_titles))
    if n < 0:
        raise ValueError("n must be non-negative")
    if target not in titles:
        raise ValueError(f"Target not in catalog: {target}")
    excluded = set(history_titles) | {target}
    pool = [title for title in titles if title not in excluded]
    if len(pool) < n:
        raise ValueError(f"Not enough negatives: need {n}, available {len(pool)}")
    candidates = rng.sample(pool, n) + [target]
    rng.shuffle(candidates)
    return candidates


def history_titles(history, catalog):
    titles = []
    for title in history.split('; '):
        if title in catalog.index:
            titles.append(title)
            continue
        # The preprocessing notebook truncates history titles to 50 characters.
        matches = sorted({t for t in catalog.index if len(title) == 50 and t.startswith(title)})
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous history title: {title}")
        titles.append(matches[0])
    return titles


def run_generation(task, prepare, parse, argv=None):
    parser = argparse.ArgumentParser(description=f"Generate {task} data with Qwen (Figure C9/C10).")
    parser.add_argument('input', type=Path, help='simulator_test_data_*.jsonl')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--catalog', type=Path, default=Path(__file__).resolve().parents[1] / 'test/steam/games.ftr')
    parser.add_argument('--model', default='qwen3.8-max-0902')
    parser.add_argument('--base-url', default=os.getenv('DASHSCOPE_BASE_URL', 'https://ws-2cen2i0clxvmjuvf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'))
    parser.add_argument('--concurrency', type=int, default=5, help='Samples per parallel batch')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    if args.max_tokens <= 0:
        parser.error('--max-tokens must be positive')
    if not args.input.stem.startswith('simulator_test_data_') and args.output is None:
        parser.error('Use simulator_test_data_*.jsonl or specify --output')
    prefix = 'retrival_review_data_' if task == 'retrival' else 'ranking_eval_data_'
    output = args.output or args.input.with_name(args.input.name.replace('simulator_test_data_', prefix, 1))
    audit_path = output.with_suffix('.audit.jsonl')
    for path in (output, audit_path):
        if path.exists():
            parser.error(f'Refusing to overwrite {path}')
    api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-ws-H.PXDXLYY.wRgv.MEYCIQCcvKWydSrSKH5KPQTwzixBF1IZ_uv60sjfDw8JUXDcAAIhANEdb9cSgSBfYDV2hRDrNqP565abOo_kAj5FJP7BDKZY')
    if not api_key:
        parser.error('Set DASHSCOPE_API_KEY in the environment')
    catalog = pd.read_feather(args.catalog)
    if catalog['title'].isna().any():
        parser.error('Catalog titles must be non-null')
    catalog = catalog.set_index('title')
    lines = args.input.read_text(encoding='utf-8').splitlines()
    if not lines:
        parser.error('Input is empty')
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120, max_retries=2)
    def generate(index, line):
        result = None
        record = {'line': index + 1, 'model': args.model, 'base_url': args.base_url,
                  'seed': args.seed, 'temperature': args.temperature,
                  'max_tokens': args.max_tokens, 'enable_thinking': False,
                  'generation_status': 'input_error', 'raw_input': line}
        fatal = False
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('history', 'target')):
                raise ValueError('Expected non-empty history and target strings')
            record['input'] = row
            if row['target'] not in catalog.index:
                raise ValueError('Target is absent from catalog')
            history = history_titles(row['history'], catalog)
            prompt, details = prepare(row, catalog, history, random.Random(args.seed + index))
            record.update(prompt=prompt, details=details)
            messages = [{'role': 'user', 'content': prompt}]
            if details.get('system_prompt'):
                messages.insert(0, {'role': 'system', 'content': details['system_prompt']})
            options = {'response_format': details['response_format']} if 'response_format' in details else {}
            record['generation_status'] = 'api_error'
            response = client.chat.completions.create(
                model=args.model, messages=messages,
                temperature=args.temperature, max_tokens=args.max_tokens,
                extra_body={'enable_thinking': False}, **options,
            )
            record['response'] = response.model_dump(mode='json')
            record['generation_status'] = 'empty'
            if not response.choices:
                raise ValueError('No response choices')
            choice = response.choices[0]
            record['raw_response'] = choice.message.content
            if choice.message.refusal or choice.finish_reason == 'content_filter':
                record['generation_status'] = 'refused'
            elif choice.finish_reason != 'stop':
                record['generation_status'] = 'incomplete'
            elif choice.message.content:
                record['generation_status'] = 'complete'
            if record['generation_status'] != 'complete':
                raise ValueError(f'Incomplete, refused or empty response: {choice.finish_reason}')
            value = parse(choice.message.content, row, history, details)
            if task == 'retrival':
                record['candidate'] = value
            else:
                result = {'question': value, 'target': row['target']}
            record['status'] = 'valid'
        except (ValueError, TypeError, KeyError, OpenAIError) as exc:
            record.update(status='failed', error=str(exc), error_type=type(exc).__name__)
            record['error_causes'] = []
            cause = exc.__cause__
            seen = {id(exc)}
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                record['error_causes'].append({'type': type(cause).__name__,
                                               'message': str(cause).replace(api_key, '[REDACTED]')})
                cause = cause.__cause__
            if isinstance(exc, APIStatusError):
                record['api_error'] = exc.body
                # Policy rejection is sample-specific; auth/model/quota errors are not.
                code = json.dumps(exc.body, ensure_ascii=False).lower()
                policy = any(word in code for word in ('inappropriate', 'content_filter', 'content_policy', 'data_inspection', 'safety', 'sensitive'))
                fatal = not policy
            elif isinstance(exc, OpenAIError):
                fatal = True
        if task == 'retrival':
            candidate = record.get('candidate', {})
            flags = list(candidate.get('review_flags', []))
            if 'error' in record:
                flags.append(record['error'])
            result = {
                'line': index + 1, 'total_samples': len(lines), 'input': record.get('input'),
                'target': (record.get('input') or {}).get('target'),
                'generation_status': record['generation_status'],
                'id_to_title': record.get('details', {}).get('id_to_title', {}),
                'target_info': record.get('details', {}).get('target_info'),
                'raw_response': record.get('raw_response'),
                'messages': candidate.get('messages'), 'context': candidate.get('context'),
                'review_flags': flags, 'review_status': 'pending', 'review_notes': '',
            }
            record['status'] = 'pending_review'
        return record, result, fatal

    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with client, output.open('x', encoding='utf-8') as dest, audit_path.open('x', encoding='utf-8') as audit, ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for start in range(0, len(lines), args.concurrency):
            futures = [executor.submit(generate, index, lines[index])
                       for index in range(start, min(start + args.concurrency, len(lines)))]
            batch = [future.result() for future in futures]
            fatal_batch = False
            for record, result, fatal in batch:
                if result is not None:
                    dest.write(json.dumps(result, ensure_ascii=False) + '\n')
                    dest.flush()
                    written += 1
                audit.write(json.dumps(record, ensure_ascii=False) + '\n')
                audit.flush()
                fatal_batch |= fatal
                print(f'{record["line"]}/{len(lines)}: {record["status"]}; saved={written}', flush=True)
            if fatal_batch:
                raise SystemExit(f'API failure; stopped after saving the batch. Details: {audit_path}')
    label = 'review records (not evaluation-ready)' if task == 'retrival' else 'valid samples'
    print(f'Saved {written}/{len(lines)} {label} to {output}; audit: {audit_path}')
    if not written:
        raise SystemExit('No samples saved')
