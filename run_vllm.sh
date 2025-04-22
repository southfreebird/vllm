vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
          --served-model-name meta-llama/Meta-Llama-3.1-8B-Instruct-fast \
          --trust-remote-code \
          --disable-log-requests \
          --tensor-parallel-size 2 \
          --gpu-memory-utilization 0.97 \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser llama3_json \
          --speculative-config '{"model": "./test-medusa/", "num_speculative_tokens": 1, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "method": "medusa"}'
        #   --speculative-config '{"model": "./test-medusa/", "num_speculative_tokens": 3, "disable_logprobs": false, "disable_log_stats": false, "disable_by_batch_size": 30, "draft_tensor_parallel_size": 1}'