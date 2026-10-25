"""Offline checks for the replay launcher and reflection-history preservation."""
import ast
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest.mock import patch, MagicMock

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('replay', here / 'run.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
with TemporaryDirectory() as tmp, patch.object(replay, 'HERE', Path(tmp)):
    with patch.object(replay.socket, 'create_connection', return_value=MagicMock()), \
         patch.object(replay.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
        assert replay.launch() == 0
        script = run.call_args.kwargs['input']
        assert '--worker --save=' in script
        assert 'FRAMEWORK_DIR=' + str(replay.FRAMEWORK) in script
        assert len((Path(tmp) / 'input.jsonl').read_text().splitlines()) == 40
        source = replay.FRAMEWORK / 'test/steam/retrieval_eval_data_500.jsonl'
        assert (Path(tmp) / 'input.jsonl').read_bytes() == b''.join(source.read_bytes().splitlines(keepends=True)[:40])
        assert not any(k.lower() in {'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'} for k in run.call_args.kwargs['env'])
    with patch.object(replay.socket, 'create_connection', side_effect=OSError('offline')), \
         patch.object(replay.subprocess, 'run') as run:
        assert replay.worker() == 2
        run.assert_not_called()
print('PASS: original first 40 rows, launcher substitutions, proxy removal, offline fail-fast')

path = replay.FRAMEWORK / 'planning/agent_plan_first_openai.py'
tree = ast.parse(path.read_text())
cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'CRSAgentPlanFirstOpenAI')
fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'run')
for arg in fn.args.args:
    arg.annotation = None
fn.returns = None
scope = {'logger': SimpleNamespace(debug=lambda *a: None)}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
agent = SimpleNamespace(
    selector=None, toolbox=SimpleNamespace(name='tools', desc='', failed_times=0),
    item_corups=SimpleNamespace(info=lambda **kw: ''),
    candidate_buffer=SimpleNamespace(clear_tracks=lambda: None, clear=lambda: None, track_info=''),
    memory=SimpleNamespace(get=lambda: '', append=lambda *a: None),
    _record_planning=False, prompt='', _reflection_cnt=0, reflection_limits=3,
    _k_turn=0, user_profile_update=-1)
histories = []
agent.plan_and_exe = lambda prompt, values: histories.append(values['history']) or 'five recommendations'
agent.critic = lambda *args: (len(histories) == 1, 'try again')
agent.run = MethodType(scope['run'], agent)
agent.run({'input': 'Please give 5 recommendations'}, chat_history='USER_HISTORY_SENTINEL')
assert histories == ['USER_HISTORY_SENTINEL', 'USER_HISTORY_SENTINEL'], histories
print('PASS: reflection preserves complete chat history')
