#!/bin/bash
set -euo pipefail

FRAMEWORK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$FRAMEWORK_DIR"

# Optional local credentials; excluded from Git.
if [[ -f "$FRAMEWORK_DIR/run.local.env" ]]; then
    source "$FRAMEWORK_DIR/run.local.env"
fi

# API related info
API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY or API_KEY}}" # your api key
API_BASE="${API_BASE:-${OPENAI_API_BASE:?Set OPENAI_API_BASE or API_BASE}}" # [https://xxxx.azure.com, https://api.openai.com/v1]
API_TYPE="open_ai" # ['open_ai', 'azure']
engine="deepseek-flash"   # model name for OpenAI or deployment name for Azure OpenAI. GPT-4 is recommended.
bot_type="chat" # model type, ["chat", "completetion"]. For gpt-3.5-turbo and gpt-4, it should be "chat". For text-davinci-003, it should be "completetion" 


domain="game"   # item domain
enable_shorten=0    # whether to enable shorten chat history

# demonstration mode. 
# 1. zero: Zero-shot setting. No demonstration would be inserted into prompt.
# 2. fixed: Fixed demonstrations are used for all cases. It would use the first n demonstrations in the `demo_dir_or_file`, where n is `num_demos`.
# 3. dynamic: Retrieval the most n relevant demonstrations.
demo_mode="dynamic"  # ["zero", "fixed", "dynamic"]

num_demos=5 # number of demonstrations to use, when demo_mode=="fixed" and num_demos<0, all demonstrations would be used.

# folder or file path of demonstrations. If folder, all jsonl files would be used. 
# If demo_mode=="zero", the argument does not function.
demo_dir_or_file="./resources/seed_demos_placeholder.jsonl" 


enable_reflection=0 # whether to use reflection. Reflection would increase the token usage and the response delay.


DOMAIN=$domain \
OPENAI_API_KEY=$API_KEY \
OPENAI_API_BASE=$API_BASE \
OPENAI_API_TYPE=$API_TYPE \
TOKENIZERS_PARALLELISM=false \
"$FRAMEWORK_DIR/../.recai/bin/python" -u app.py \
    --engine=$engine \
    --bot_type=$bot_type \
    --enable_shorten=$enable_shorten \
    --demo_mode=$demo_mode \
    --num_demos=$num_demos \
    --enable_reflection=$enable_reflection \
    --demo_dir_or_file=$demo_dir_or_file
