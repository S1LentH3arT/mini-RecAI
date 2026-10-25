import unittest
from threading import Barrier, Lock
from types import SimpleNamespace

from parallel_eval import batch_answers, agent_cloner, check_agent_resources
from unittest.mock import patch


class BatchTests(unittest.TestCase):
    def test_five_independent_sessions_and_order(self):
        barrier = Barrier(5, timeout=5)
        lock = Lock()
        active = 0
        maximum = 0
        completed = []
        workers = []

        class Worker:
            def __init__(self):
                self.history = []
                workers.append(self)

            def run(self, value):
                nonlocal active, maximum
                self.history.append(value)
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    if value >= 5:
                        assert len(completed) >= 5
                barrier.wait()
                with lock:
                    completed.append(value)
                    active -= 1
                return str(value)

        rows = [{'context': i} for i in range(10)]
        with patch('parallel_eval.agent_cloner', return_value=Worker):
            result = list(batch_answers(rows, SimpleNamespace(bot=object()), 'context'))
        self.assertEqual(maximum, 5)
        self.assertEqual([r[2] for r in result], list(map(str, range(10))))
        self.assertEqual(len(workers), 10)
        self.assertTrue(all(len(w.history) == 1 for w in workers))

    def test_clone_graph_keeps_resources_and_rebinds_state(self):
        class Buffer:
            def __init__(self):
                self.memory = []
            def init_candidates(self, value):
                self.memory.append(value)
        class Gallery:
            def info(self, **kwargs):
                return 'catalog'
        class Toolbox:
            def __init__(self, tools):
                self.tools = tools
            def run(self, value):
                self.tools[0].func(value)
        buffer = Buffer()
        tool = SimpleNamespace(buffer=buffer, model=object(), item_sim=object())
        store = SimpleNamespace(func=buffer.init_candidates)
        agent = SimpleNamespace(user_profile_update=-1, _record_planning=False,
                                item_corups=Gallery(), agent=object(), selector=None,
                                memory=SimpleNamespace(memory=[]), candidate_buffer=buffer,
                                critic=SimpleNamespace(bot=object(), buffer=buffer),
                                _tools=[store, tool], toolbox=Toolbox([store, tool]),
                                _plan_record_cache={'traj': []})
        wrapper = SimpleNamespace(bot=agent)
        report = check_agent_resources(wrapper)
        self.assertEqual(report['isolation'], 'passed')
        clone = agent_cloner(wrapper)
        one, two = clone(), clone()
        one.bot.toolbox.run('private')
        one.bot.memory.memory.append('private')
        one.bot._plan_record_cache['traj'].append('private')
        self.assertEqual(one.bot.candidate_buffer.memory, ['private'])
        self.assertEqual(two.bot.candidate_buffer.memory, [])
        self.assertEqual(two.bot.memory.memory, [])
        self.assertEqual(two.bot._plan_record_cache['traj'], [])
        self.assertEqual(buffer.memory, [])

    def test_failure_remains_in_results(self):
        class Broken:
            def run(self, value):
                raise ValueError('bad')
        result = list(batch_answers([{'question': 'x'}], Broken(), 'question', 1))
        self.assertEqual(len(result), 1)
        self.assertIn('ValueError', result[0][2])

    def test_metrics_match_serial_with_partial_last_batch(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from eval import one_turn_eval, one_turn_eval_ranking
        class Echo:
            bot = object()
            def run(self, value):
                return value
        retrieval = [{'context': 'Target' if i % 2 else 'Other', 'target': 'Target'} for i in range(7)]
        ranking = [{'question': '{"Target": %d}' % (i + 1), 'target': 'Target'} for i in range(7)]
        with patch('eval.parallel_eval.agent_cloner', return_value=Echo):
            serial = one_turn_eval.one_turn_conversation_eval(retrieval, Echo(), 1)
            parallel = one_turn_eval.one_turn_conversation_eval(retrieval, Echo(), 5)
            self.assertEqual(serial, parallel)
            serial = one_turn_eval_ranking.one_turn_conversation_eval(ranking, Echo(), 20, 1)
            parallel = one_turn_eval_ranking.one_turn_conversation_eval(ranking, Echo(), 20, 5)
            self.assertEqual(serial, parallel)

    def test_review_file_cannot_be_used_as_evaluation_input(self):
        import json
        import sys
        import tempfile
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from eval.one_turn_eval import read_jsonl
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.jsonl'
            path.write_text(json.dumps({'context': 'preview', 'target': 'x', 'review_status': 'pending'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'export-reviewed'):
                read_jsonl(str(path))

    def test_invalid_configuration(self):
        for rows, n in [([], 1), ([{}], 0), ([{}], 5)]:
            with self.assertRaises(ValueError):
                list(batch_answers(rows, object(), 'context', n))


if __name__ == '__main__':
    unittest.main()
