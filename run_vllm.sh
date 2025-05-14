vllm serve Qwen/Qwen3-235B-A22B-FP8 \
          --served-model-name Qwen/Qwen3-235B-A22B-FP8 \
          --trust-remote-code \
          --disable-log-requests \
          --tensor-parallel-size 4 \
          --gpu-memory-utilization 0.97 \
          --enable-auto-tool-choice \
          --tool-call-parser llama3_json \
          --seed 42 \
        #   --speculative-config '{"model": "/papyrax/from_Qwen3_235A22/mlp_based_lk/", "num_speculative_tokens": 2, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "method": "mlp_speculator"}'
          # --speculative-config '{"model": "./from_llama_v3.1_8b_instruct/medusa_lk/", "num_speculative_tokens": 3, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "method": "medusa"}'
          # --speculative-config '{"model": "./from_llama_v3.1_8b_instruct/mlp_based_lk/", "num_speculative_tokens": 3, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "method": "mlp_speculator"}'

