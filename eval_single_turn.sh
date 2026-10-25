#!/bin/bash
set -euo pipefail

FRAMEWORK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$FRAMEWORK_DIR"

# Optional local credentials; excluded from Git.
if [[ -f "$FRAMEWORK_DIR/eval_single_turn.local.env" ]]; then
    source "$FRAMEWORK_DIR/eval_single_turn.local.env"
fi

# API related info
API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY or API_KEY}}" # your api key
API_BASE="${API_BASE:-${OPENAI_API_BASE:?Set OPENAI_API_BASE or API_BASE}}" # [https://xxxx.azure.com, https://api.openai.com/v1]
API_TYPE="open_ai" # ['open_ai', 'azure']
engine="qwen3.7-flash"   # model name for OpenAI or deployment name for Azure OpenAI. GPT-4 is recommended.
bot_type="chat" # model type, ["chat", "completetion"]. For gpt-3.5-turbo and gpt-4, it should be "chat". For text-davinci-003, it should be "completetion"
agent="recbot" # agent name, ["recbot", "recbot-critic"]. recbot-critic is a recbot with critic module. It is recommended to use recbot-critic for better performance.


domain="game"   # item domain
enable_shorten=0    # whether to enable shorten chat history

# demonstration mode. 
# 1. zero: Zero-shot setting. No demonstration would be inserted into prompt.
# 2. fixed: Fixed demonstrations are used for all cases. It would use the first n demonstrations in the `demo_dir_or_file`, where n is `num_demos`.
# 3. dynamic: Retrieval the most n relevant demonstrations.
demo_mode="dynamic"  # ["zero", "fixed", "dynamic"]
demo_dir_or_file="$FRAMEWORK_DIR/resources/seed_demos.jsonl"
num_demos=3 # number of demonstrations to use. If demo_mode=="fixed", it would use the first num_demos demonstrations in the `demo_dir_or_file`. If demo_mode=="dynamic", it would retrieve the most num_demos relevant demonstrations.

num_rec=5 # number of recommendations to generate, when demo_mode=="fixed" and num_demos<0, all demonstrations would be used.
k=20 # NDCG cutoff for the ranking evaluation.

# folder or file path of demonstrations. If folder, all jsonl files would be used. 
# If demo_mode=="zero", the argument does not function.
retrieval_data="$FRAMEWORK_DIR/test/steam/retrieval_eval_data_500.jsonl"
ranking_data="$FRAMEWORK_DIR/test/steam/one_turn_eval_ranking_500.jsonl"
source_data="$FRAMEWORK_DIR/test/steam/simulator_test_data_500.jsonl"


enable_reflection=1 # whether to use reflection. Reflection would increase the token usage and the response delay.
max_candidate_num=1000 # max number of candidates. If the number of candidates is larger than this value, only the top max_candidate_num candidates would be used for reflection.
similar_ratio=0.05 # similarity ratio. If the similarity between the candidate and the user query is larger than this value, the candidate would be used for reflection.


timeout=180 # timeout for each API call
concurrency=5 # number of concurrent API calls. If concurrency>1, the evaluation would be faster but may cause more API errors.


PYTHON_BIN="$FRAMEWORK_DIR/../.recai/bin/python" # python binary path


retrieval_max_output_tokens=2048 # max output tokens for retrieval evaluation
ranking_max_output_tokens=2048 # max output tokens for ranking evaluation

# retrieval evaluation
DOMAIN=$domain \
OPENAI_API_KEY=$API_KEY \
OPENAI_API_BASE=$API_BASE \
OPENAI_API_TYPE=$API_TYPE \
AGENT_ENGINE=$engine \
TOKENIZERS_PARALLELISM=false \
$PYTHON_BIN -u "$FRAMEWORK_DIR/eval/one_turn_eval.py" \
    --max_output_tokens=$retrieval_max_output_tokens \
	--agent=$agent \
    --engine=$engine \
    --bot_type=$bot_type \
	--data=$retrieval_data \
	--num_rec=$num_rec \
	--timeout=$timeout \
    --enable_shorten=$enable_shorten \
    --demo_mode=$demo_mode \
    --demo_dir_or_file=$demo_dir_or_file \
    --num_demos=$num_demos \
    --enable_reflection=$enable_reflection \
    --max_candidate_num=$max_candidate_num \
    --concurrency=$concurrency \
    --similar_ratio=$similar_ratio

# ranking evaluation
# TOKENIZERS_PARALLELISM=false \
# $PYTHON_BIN -u "$FRAMEWORK_DIR/eval/one_turn_eval_ranking.py" \
#     --max_output_tokens=$ranking_max_output_tokens \
# 	--agent=$agent \
#     --engine=$engine \
#     --bot_type=$bot_type \
# 	--data=$ranking_data \
# 	--k=$k \
# 	--timeout=$timeout \
#     --max_output_tokens=2048 \
#     --enable_shorten=$enable_shorten \
#     --demo_mode=$demo_mode \
#     --enable_reflection=$enable_reflection 

# import os
# from openai import OpenAI

# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
#     api_key="sk-ws-H.PXDXLYY.wRgv.MEYCIQCcvKWydSrSKH5KPQTwzixBF1IZ_uv60sjfDw8JUXDcAAIhANEdb9cSgSBfYDV2hRDrNqP565abOo_kAj5FJP7BDKZY",
#     base_url="https://ws-2cen2i0clxvmjuvf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
# )

# completion = client.chat.completions.create(
#     # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     model="qwen3.8-flash",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你是谁？"},
#     ]
# )
# print(completion.model_dump_json())
