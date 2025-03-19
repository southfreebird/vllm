# SPDX-License-Identifier: Apache-2.0
num_spec_tokens = 5


def find_last_match(last_n_tokens: list[int], input_tokens: list[int]):
    n = len(last_n_tokens)
    inp_seq_n = len(input_tokens)

    if n == 0:
        return 0

    for match_len in range(num_spec_tokens + n, 0, -1):
        sub_last_n_tokens = last_n_tokens[:match_len]
        last_idx = min(inp_seq_n, inp_seq_n - match_len + n)
        sub_input_tokens = input_tokens[-match_len:last_idx]

        print(sub_last_n_tokens, sub_input_tokens)
        if sub_input_tokens == sub_last_n_tokens:
            # return n - match_len
            print("ans:", n - match_len)

    # raise ValueError("Number of rollback tokens exceed max_rollback_tokens")


last_n = (788, 7342, 330, 1313, 62)
input_tokens = (288, 788, 7342, 330, 1313, 62, 30775, 72, 968, 788)
print("Result:", find_last_match(last_n, input_tokens))  # Expected: 0
