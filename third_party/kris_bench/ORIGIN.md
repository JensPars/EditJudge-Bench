The KRIS-Bench evaluation prompts in `editjudge/prompts/kris_bench.py` are
copied verbatim from https://github.com/mercurystraw/Kris_Bench
(`utils/prompts.py` at commit cf1475253a13f5e5666826c10d8967e38917468a,
Apache-2.0, see LICENSE). Only `prompt_instruction_following` and
`prompt_consist` are included, unmodified. The message layout (prompt text,
then "This is the original image:" + original image, then "This is the edited
image:" + edited image, no system prompt) follows `evaluate_with_gpt` in
`metrics_common.py` at the same commit.
