"""Five-sample batches with isolated Agent state and one set of local models."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextvars import Context
from functools import wraps
from threading import RLock


def locked_call(func, lock):
    @wraps(func)
    def call(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    return call


def agent_cloner(wrapper):
    """Keep the initialized template pristine; clone session state, not weights."""
    agent = wrapper.bot
    if agent.user_profile_update > 0 or agent._record_planning:
        raise ValueError('Parallel one-turn evaluation requires long-term memory and plan-file recording disabled')
    lock = RLock()
    shared = [agent.item_corups, agent.agent, agent.selector,
              getattr(agent.memory, '_shortening_bot', None)]
    if agent.critic is not None:
        shared.append(agent.critic.bot)
    for tool in agent._tools:
        shared.extend(getattr(tool, key, None) for key in ('model', 'item_sim'))
    memo = {id(value): value for value in shared if value is not None}
    # Serial local inference bounds scratch memory; API waits remain parallel.
    agent.item_corups.info = locked_call(agent.item_corups.info, lock)
    selector = locked_call(agent.selector, lock) if agent.selector else None

    def clone():
        worker = deepcopy(wrapper, memo.copy())
        worker.bot.selector = selector
        worker.bot.toolbox.run = locked_call(worker.bot.toolbox.run, lock)
        return worker
    return clone


def batch_answers(data, agent, field, concurrency=5):
    if concurrency < 1:
        raise ValueError('concurrency must be positive')
    if not data:
        raise ValueError('Evaluation data is empty')
    if concurrency > 1 and not hasattr(agent, 'bot'):
        raise ValueError('Parallel evaluation currently supports recbot only; use --concurrency=1 for baselines')
    clone = agent_cloner(agent) if concurrency > 1 else lambda: agent

    def run(worker, row):
        try:
            if hasattr(worker, 'clear'):
                worker.clear()
            return worker.run(row[field])
        except Exception as exc:
            # Retain failures in the denominator and in the saved transcript.
            return f'Experiment execution failed: {type(exc).__name__}'

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for start in range(0, len(data), concurrency):
            batch = data[start:start + concurrency]
            workers = [clone() for _ in batch]
            futures = [executor.submit(Context().run, run, worker, row) for worker, row in zip(workers, batch)]
            # Wait for the entire batch; write and score in original input order.
            answers = [future.result() for future in futures]
            for offset, (row, answer) in enumerate(zip(batch, answers)):
                yield start + offset, row, answer


def check_agent_resources(wrapper, concurrency=5):
    """Offline initialization check: no remote calls and no model duplication."""
    from pathlib import Path

    def memory():
        status = Path('/proc/self/status')
        if not status.exists():
            return {}
        return {line.split(':')[0]: int(line.split()[1]) / 1024
                for line in status.read_text().splitlines()
                if line.startswith(('VmRSS:', 'VmHWM:'))}

    before = memory()
    clone = agent_cloner(wrapper)
    workers = [clone() for _ in range(concurrency)]
    for attr in ('candidate_buffer', 'memory', 'toolbox', '_plan_record_cache'):
        assert len({id(getattr(w.bot, attr)) for w in workers}) == concurrency, attr
    for worker in workers:
        agent = worker.bot
        assert agent.item_corups is wrapper.bot.item_corups
        if agent.critic:
            assert agent.critic is not wrapper.bot.critic
            assert agent.critic.buffer is agent.candidate_buffer
        for tool, original in zip(agent._tools, wrapper.bot._tools):
            assert tool is not original
            if hasattr(tool, 'buffer'):
                assert tool.buffer is agent.candidate_buffer
            if hasattr(tool, 'func'):
                assert tool.func.__self__ is agent.candidate_buffer
            for attr in ('model', 'item_sim'):
                if hasattr(tool, attr):
                    assert getattr(tool, attr) is getattr(original, attr)
    return {'workers': concurrency, 'isolation': 'passed',
            'loaded_MiB': before, 'with_workers_MiB': memory()}
