"""Replay the source script on exactly its first 40 input rows, with observation only."""
import contextvars
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parents[2]


def launch():
    source = FRAMEWORK / 'test/steam/retrieval_eval_data_500.jsonl'
    prefix = b''.join(source.read_bytes().splitlines(keepends=True)[:40])
    assert len(prefix.splitlines()) == 40
    (HERE / 'input.jsonl').write_bytes(prefix)
    script = (FRAMEWORK / 'eval_single_turn.sh').read_text()
    assert script.count('$PYTHON_BIN -u "$FRAMEWORK_DIR/eval/one_turn_eval.py"') == 1
    script = re.sub(r'^FRAMEWORK_DIR=.*$', 'FRAMEWORK_DIR=' + shlex.quote(str(FRAMEWORK)), script, count=1, flags=re.M)
    script = re.sub(r'^retrieval_data=.*$', 'retrieval_data=' + shlex.quote(str(HERE / 'input.jsonl')), script, count=1, flags=re.M)
    script = script.replace('$PYTHON_BIN -u "$FRAMEWORK_DIR/eval/one_turn_eval.py"',
                            '$PYTHON_BIN -u ' + shlex.quote(str(Path(__file__).resolve()))
                            + ' --worker --save=' + shlex.quote(str(HERE / 'answers.jsonl')))
    manifest = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'rows': 40,
                'script_sha256': hashlib.sha256((FRAMEWORK / 'eval_single_turn.sh').read_bytes()).hexdigest(),
                'model': re.search(r'^engine="([^"]+)"', script, re.M).group(1),
                'started': time.time(), 'status': 'running'}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    env = {k: v for k, v in os.environ.items() if k.lower() not in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'}}
    with (HERE / 'run.log').open('w') as log:
        result = subprocess.run(['bash'], input=script, text=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(exit_code=result.returncode, finished=time.time(),
                    status='completed' if result.returncode == 0 else 'failed')
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return result.returncode


def worker():
    # Shell configuration and optional local credentials have been resolved here.
    host = urlparse(os.environ.get('OPENAI_API_BASE', '')).hostname
    try:
        with socket.create_connection((host, 443), timeout=10):
            pass
    except OSError as exc:
        print('API connection unavailable; evaluation not started:', exc)
        return 2
    sys.path.insert(0, str(FRAMEWORK))
    from eval import one_turn_eval as evaluator
    from tools import SQLSearchTool, SimilarItemTool
    from tools.reco_model_tool import RecModelTool
    from tools.map_tool import MapTool
    from utils.open_ai import OpenAICall
    from planning.agent_plan_first_openai import CRSAgentPlanFirstOpenAI, ToolBox

    rows = evaluator.read_jsonl(str(HERE / 'input.jsonl'))
    sample = contextvars.ContextVar('sample', default=None)
    lock = threading.Lock()
    trace = (HERE / 'trace.jsonl').open('w')

    def emit(event, **data):
        with lock:
            trace.write(json.dumps({'sample': sample.get(), 'event': event, **data}, default=str) + '\n')
            trace.flush()

    original_run = evaluator.RecBotWrapper.run

    def run(self, history):
        indices = [i for i, row in enumerate(rows) if row['context'] == history]
        assert len(indices) == 1, 'Ambiguous sample identity'
        token = sample.set(indices[0] + 1)
        emit('start', target=rows[indices[0]]['target'], demos_enabled=self.bot.selector is not None)
        try:
            answer = original_run(self, history)
            emit('answer', answer=answer, hit=evaluator.hit_judge(answer, rows[indices[0]]['target']))
            return answer
        except Exception:
            emit('error', traceback=traceback.format_exc())
            raise
        finally:
            sample.reset(token)

    evaluator.RecBotWrapper.run = run

    def observe(cls, name, tool=False):
        original = getattr(cls, name)

        @functools.wraps(original)
        def observed(self, *args, **kwargs):
            before = list(self.buffer.get()) if tool else None
            emit('call', component=cls.__name__, method=name, args=args, kwargs=kwargs, candidates_before=before)
            try:
                result = original(self, *args, **kwargs)
                emit('return', component=cls.__name__, method=name, result=result,
                     candidates_after=list(self.buffer.get()) if tool else None)
                return result
            except Exception:
                emit('call_error', component=cls.__name__, method=name, traceback=traceback.format_exc())
                raise
        setattr(cls, name, observed)

    for cls in (SQLSearchTool, SimilarItemTool, RecModelTool, MapTool):
        observe(cls, 'run', tool=True)
    observe(OpenAICall, 'call')
    observe(ToolBox, 'run')
    observe(CRSAgentPlanFirstOpenAI, 'plan_and_exe')
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    try:
        evaluator.main()
    finally:
        trace.close()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        raise SystemExit(worker())
    else:
        raise SystemExit(launch())
