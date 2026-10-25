#!/bin/bash
set -euo pipefail

FRAMEWORK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$FRAMEWORK_DIR"

# Optional local credentials; excluded from Git.
if [[ -f "$FRAMEWORK_DIR/eval_simulator.local.env" ]]; then
    source "$FRAMEWORK_DIR/eval_simulator.local.env"
fi

PYTHON_BIN="$FRAMEWORK_DIR/../.recai/bin/python"

max_output_tokens=2048
simulator_max_output_tokens=1024 


# - environment variables start with "AGENT_" or "OPENAI_" are used for conversational agent
# - environment variables start with "SIMULATOR_" are used for user simulator 


#!/bin/bash  
DOMAIN="game" \
AGENT_ENGINE="deepseek-flash" \
OPENAI_API_KEY="${OPENAI_API_KEY:-${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY or OPENAI_API_KEY}}" \
OPENAI_API_BASE="${OPENAI_API_BASE:-${API_BASE:?Set API_BASE or OPENAI_API_BASE}}" \
OPENAI_API_TYPE="open_ai" \
SIMULATOR_ENGINE="qwen3.8-max-0902" \
SIMULATOR_API_KEY="${SIMULATOR_API_KEY:-${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY or SIMULATOR_API_KEY}}" \
SIMULATOR_API_BASE="${SIMULATOR_API_BASE:-${OPENAI_API_BASE:?Set OPENAI_API_BASE or SIMULATOR_API_BASE}}" \
SIMULATOR_API_TYPE="open_ai" \
SIMULATOR_ENGINE_TYPE="chat_completion" \
TOKENIZERS_PARALLELISM=false  \
$PYTHON_BIN -u "$FRAMEWORK_DIR/eval/user_simulator.py" \
    --simulator_max_output_tokens=$max_output_tokens \
    --max_output_tokens=$max_output_tokens \
    --timeout=60 \
    --bot_type="chat" \
    --enable_shorten=0 \
    --engine="deepseek-flash" \
    --num_demos=5 \
    --enable_reflection=1 \
    --data="$FRAMEWORK_DIR/test/steam/simulator_test_data_10.jsonl" \
    --max_turns=5 \
    --agent="recbot"
