"""Offline tests: no model weights, credentials or network required."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import data_generation as common
import raw_2_ranking_data as ranking
import raw_2_retrival_data as retrieval


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.row = {'history': 'Old Game', 'target': 'Desired Game'}
        self.history = ['Old Game']
        titles = self.history + ['Desired Game'] + [f'Negative {i:02d}' for i in range(25)]
        self.catalog = pd.DataFrame({'title': titles, 'release_date': ['2020'] * len(titles),
                                     'tags': ['Action'] * len(titles), 'price': [5] * len(titles),
                                     'description': ['A game'] * len(titles)}).set_index('title')
        self.dialogue = json.dumps({'messages': [
            {'role': 'User', 'text': 'I played [H1].', 'mentioned_ids': ['H1']},
            {'role': 'Assistent', 'text': 'What would you like next?', 'mentioned_ids': []},
            {'role': 'User', 'text': 'An action game priced around five.', 'mentioned_ids': []},
            {'role': 'Assistent', 'text': 'Try <item>.', 'mentioned_ids': []},
        ]})

    def test_sampling_and_ranking_contract(self):
        prompt, details = ranking.prepare(self.row, self.catalog, self.history, random.Random(2024))
        candidates = details['candidates']
        self.assertEqual(len(set(candidates)), 20)
        self.assertEqual(candidates.count(self.row['target']), 1)
        self.assertNotIn('Old Game', candidates)
        self.assertEqual(details, ranking.prepare(self.row, self.catalog, self.history, random.Random(2024))[1])
        question = 'I played Old Game. Please rank: ' + '; '.join(candidates)
        self.assertEqual(ranking.parse(question, self.row, self.history, details), question)
        with self.assertRaises(ValueError):
            ranking.parse(question.replace('Desired Game', 'Other Game'), self.row, self.history, details)
        with self.assertRaises(ValueError):
            common.neg_item_sample(['Desired Game'], [], 'Desired Game', 19, random.Random(1))
        self.assertIn('20', prompt)

    def test_retrieval_contract_and_leakage(self):
        prompt, details = retrieval.prepare(self.row, self.catalog, self.history, random.Random(1))
        self.assertEqual(set(details['target_info']), {'release_date', 'tags', 'price', 'description'})
        self.assertNotIn('Desired Game', prompt)
        self.assertIn('H1 = Old Game', prompt)
        candidate = retrieval.parse(self.dialogue, self.row, self.history, details)
        self.assertTrue(candidate['context'].endswith('User: An action game priced around five.'))
        self.assertNotIn('<item>', candidate['context'])
        self.assertIn('Old Game', candidate['context'])
        self.assertEqual(candidate['messages'][0]['mentioned_items'], ['Old Game'])
        self.assertEqual(candidate['review_flags'], [])
        leaked = retrieval.parse(self.dialogue.replace('action game', 'Desired Game'), self.row, self.history, details)
        self.assertTrue(any('target-title' in f for f in leaked['review_flags']))
        premature = retrieval.parse(self.dialogue.replace('action game', '<item>'), self.row, self.history, details)
        self.assertIsNone(premature['context'])
        self.assertIn('Premature target placeholder', premature['review_flags'])
        for bad in ('{}', 'not JSON', self.dialogue.replace('"User"', '42')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                retrieval.parse(bad, self.row, self.history, details)

    def test_reference_restoration_is_exact_and_not_substring_leakage(self):
        history = ['Thief II  The Metal Age'] + [f'Other {i}' for i in range(2, 10)] + ['Game with \\ and "quotes"']
        mapping = {f'H{i + 1}': title for i, title in enumerate(history)}
        obj = json.loads(self.dialogue)
        obj['messages'][0].update(text='I played [H10], [H1], and [H10].', mentioned_ids=['H10', 'H1'])
        result = retrieval.parse(json.dumps(obj), {'target': 'Thief'}, history, {'id_to_title': mapping})
        self.assertEqual(result['messages'][0]['text'], f'I played {history[-1]}, {history[0]}, and {history[-1]}.')
        self.assertEqual(result['review_flags'], [])
        obj['messages'][0]['mentioned_ids'] = ['H1']
        result = retrieval.parse(json.dumps(obj), {'target': 'Thief'}, history, {'id_to_title': mapping})
        self.assertTrue(any('mismatch' in flag for flag in result['review_flags']))
        obj['messages'][0]['text'] = 'I played [H999].'
        with self.assertRaisesRegex(ValueError, 'Unknown history reference'):
            retrieval.parse(json.dumps(obj), {'target': 'Thief'}, history, {'id_to_title': mapping})

    def test_export_requires_review_and_rebuilds_context(self):
        _, details = retrieval.prepare(self.row, self.catalog, self.history, random.Random(1))
        candidate = retrieval.parse(self.dialogue, self.row, self.history, details)
        record = {'line': 1, 'total_samples': 1, 'input': self.row, 'target': self.row['target'],
                  'generation_status': 'complete', 'review_status': 'pending', 'review_notes': '', **candidate}
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'review.jsonl', Path(directory) / 'eval.jsonl'
            def write():
                source.write_text(json.dumps(record) + '\n')
            write()
            with self.assertRaisesRegex(ValueError, 'approved/rejected'):
                retrieval.export_reviewed(source, output)
            self.assertFalse(output.exists())
            record.update(review_status='approved', review_notes='Manually checked; formatting only.', context='stale preview')
            write()
            retrieval.export_reviewed(source, output)
            self.assertEqual(json.loads(output.read_text()), {'context': candidate['context'], 'target': self.row['target']})
            with self.assertRaises(FileExistsError):
                retrieval.export_reviewed(source, output)
            output.unlink()
            record['target'] = 'Changed label'
            write()
            with self.assertRaisesRegex(ValueError, 'original target'):
                retrieval.export_reviewed(source, output)
            record['target'] = self.row['target']
            record['generation_status'] = 'refused'
            write()
            with self.assertRaisesRegex(ValueError, 'cannot be approved'):
                retrieval.export_reviewed(source, output)
            record['generation_status'] = 'complete'
            record['total_samples'] = 2
            write()
            with self.assertRaisesRegex(ValueError, 'full set'):
                retrieval.export_reviewed(source, output)
            self.assertFalse(output.exists())

    def test_truncated_history_resolution(self):
        name = 'A' * 50 + ' full title'
        catalog = pd.DataFrame(index=[name, 'Other'])
        self.assertEqual(common.history_titles('A' * 50, catalog), [name])
        with self.assertRaises(ValueError):
            common.history_titles('A' * 50, pd.DataFrame(index=[name, 'A' * 50 + ' second']))

    def test_duplicate_titles_and_invalid_roles(self):
        catalog = pd.concat([self.catalog, self.catalog.loc[['Desired Game']]])
        with self.assertRaises(ValueError):
            retrieval.prepare(self.row, catalog, self.history, random.Random(1))
        _, details = ranking.prepare(self.row, catalog, self.history, random.Random(1))
        self.assertEqual(len(set(details['candidates'])), 20)
        with self.assertRaises(ValueError):
            retrieval.parse(self.dialogue.replace('"User"', '42'), self.row, self.history, {'id_to_title': {'H1': 'Old Game'}})

    def test_generation_five_at_a_time(self):
        from threading import Barrier, Lock
        barrier = Barrier(5, timeout=5)
        lock = Lock()
        completed = 0
        def response(**kwargs):
            nonlocal completed
            barrier.wait()
            with lock:
                completed += 1
            return self.response(self.dialogue)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'simulator_test_data_10.jsonl'
            source.write_text('\n'.join([json.dumps(self.row)] * 10))
            catalog = root / 'games.ftr'
            self.catalog.reset_index().to_feather(catalog)
            with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'offline-test'}), patch.object(common, 'OpenAI') as client:
                client.return_value.chat.completions.create.side_effect = response
                common.run_generation('retrival', retrieval.prepare, retrieval.parse, [str(source), '--catalog', str(catalog)])
            self.assertEqual(completed, 10)
            audit = [json.loads(s) for s in (root / 'retrival_review_data_10.audit.jsonl').read_text().splitlines()]
            self.assertEqual([r['line'] for r in audit], list(range(1, 11)))
            self.assertTrue(all(r['status'] == 'pending_review' for r in audit))

    def test_connection_error_records_cause(self):
        import httpx
        from openai import APIConnectionError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'simulator_test_data_1.jsonl'
            source.write_text(json.dumps(self.row) + '\n')
            catalog = root / 'games.ftr'
            self.catalog.reset_index().to_feather(catalog)
            error = APIConnectionError(request=httpx.Request('POST', 'https://example.com'))
            error.__cause__ = httpx.ConnectError('DNS failure')
            with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'offline-test'}), patch.object(common, 'OpenAI') as client:
                client.return_value.chat.completions.create.side_effect = error
                with self.assertRaises(SystemExit):
                    common.run_generation('retrival', retrieval.prepare, retrieval.parse, [str(source), '--catalog', str(catalog)])
            audit = json.loads((root / 'retrival_review_data_1.audit.jsonl').read_text())
            self.assertEqual(audit['error_causes'], [{'type': 'ConnectError', 'message': 'DNS failure'}])

    def test_ranking_runner_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'simulator_test_data_1.jsonl'
            source.write_text(json.dumps(self.row) + '\n')
            catalog = root / 'games.ftr'
            self.catalog.reset_index().to_feather(catalog)
            _, details = ranking.prepare(self.row, self.catalog, self.history, random.Random(2024))
            question = 'I played Old Game. Please rank: ' + '; '.join(details['candidates'])
            with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'offline-test'}), patch.object(common, 'OpenAI') as client:
                client.return_value.chat.completions.create.return_value = self.response(question)
                common.run_generation('ranking', ranking.prepare, ranking.parse, [str(source), '--catalog', str(catalog)])
            row = json.loads((root / 'ranking_eval_data_1.jsonl').read_text())
            self.assertEqual(row, {'question': question, 'target': 'Desired Game'})

    def response(self, text, finish='stop', refusal=None):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                               message=SimpleNamespace(content=text, refusal=refusal))],
                               model_dump=lambda **kwargs: {'text': text, 'finish_reason': finish})

    def test_runner_filters_and_preserves_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'simulator_test_data_5.jsonl'
            source.write_text('\n'.join([json.dumps(self.row)] * 5))
            catalog = root / 'games.ftr'
            self.catalog.reset_index().to_feather(catalog)
            with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'offline-test'}), patch.object(common, 'OpenAI') as client:
                client.return_value.chat.completions.create.side_effect = [
                    self.response(self.dialogue), self.response('not JSON'),
                    self.response(self.dialogue, 'length'),
                    self.response('Refused', 'content_filter'),
                    self.response(self.dialogue, refusal='Policy refusal'),
                ]
                args = [str(source), '--catalog', str(catalog), '--concurrency', '1']
                common.run_generation('retrival', retrieval.prepare, retrieval.parse, args)
                output = root / 'retrival_review_data_5.jsonl'
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(r['review_status'] == 'pending' for r in rows))
                self.assertIsNotNone(rows[0]['context'])
                self.assertIsNone(rows[1]['context'])
                self.assertEqual(rows[1]['raw_response'], 'not JSON')
                self.assertEqual([r['generation_status'] for r in rows], ['complete', 'complete', 'incomplete', 'refused', 'refused'])
                audit = [json.loads(line) for line in output.with_suffix('.audit.jsonl').read_text().splitlines()]
                self.assertEqual([r['status'] for r in audit], ['pending_review'] * 5)
                self.assertIn('prompt', audit[0])
                request = client.return_value.chat.completions.create.call_args.kwargs
                self.assertEqual(request['response_format']['type'], 'json_schema')
                self.assertTrue(request['response_format']['json_schema']['strict'])
                schema = request['response_format']['json_schema']['schema']
                self.assertEqual(schema['properties']['messages']['items']['properties']['mentioned_ids']['items']['enum'], ['H1'])
                self.assertEqual(request['messages'][0]['role'], 'system')
                with self.assertRaises(SystemExit):
                    common.run_generation('retrival', retrieval.prepare, retrieval.parse, args)
                self.assertEqual(client.return_value.chat.completions.create.call_count, 5)


if __name__ == '__main__':
    unittest.main()
