vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
          --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct-fast \
          --trust-remote-code \
          --disable-log-requests \
          --tensor-parallel-size 2 \
          --gpu-memory-utilization 0.97 \
          --enable-auto-tool-choice \
          --tool-call-parser llama3_json \
          --seed 42 \
          --speculative-config '{"model": "./from_llama_v3.1_8b_instruct/mlp_based_lk_new/", "num_speculative_tokens": 3, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "method": "mlp_speculator"}'
          # --speculative-config '{"model": "./from_llama_v3.1_8b_instruct/medusa_lk/", "num_speculative_tokens": 3, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "method": "medusa"}'
          # --speculative-config '{"model": "./from_llama_v3.1_8b_instruct/mlp_based_lk/", "num_speculative_tokens": 3, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "method": "mlp_speculator"}'

