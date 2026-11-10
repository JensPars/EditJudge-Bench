Vendored from https://github.com/VectorSpaceLab/EditScore at commit 4609c5d
(Apache-2.0, see LICENSE). Only the `editscore/` package is included.

The copy used for the paper runs had local modifications to the four
`editscore/mllm_tools/{qwen25vl,qwen25vl_vllm,qwen3vl,qwen3vl_vllm}.py`
backends; they are included here and recorded in `local_changes.patch`
(diff against 4609c5d).

One further edit: the `__main__` demo block of `editscore/mllm_tools/openai.py`
contained a hard-coded API key literal upstream; it is replaced by
`os.environ["OPENAI_API_KEY"]`. The `openai` backend is not used by
EditJudge-Bench.
