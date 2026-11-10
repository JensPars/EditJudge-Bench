"""Open-weight judges on vLLM (generative scoring and binary log-prob scoring)."""
import base64
import math
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO

from PIL import Image


@lru_cache(maxsize=20000)
def jpeg_data_uri(path: str) -> str:
    """Image downscaled to at most 512x512 pixels, as a JPEG data URI (EditJudge-Bench runs)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > 512 * 512:
        s = (512 * 512 / (w * h)) ** 0.5
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _messages(ex: dict, url) -> list:
    content = [{"type": "text", "text": v} if k == "text" else {"type": "image_url", "image_url": {"url": url(ex[v])}}
               for k, v in ex["parts"]]
    return ([{"role": "system", "content": ex["system"]}] if ex["system"] else []) + [{"role": "user", "content": content}]


def _token_id(model: str, token: str) -> int:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model).encode(token, add_special_tokens=False)[-1]


def run(cfg: dict, examples: list) -> list:
    """Raw output and score per example."""
    from vllm import LLM, SamplingParams

    model = cfg["model"]
    ejb = cfg["benchmark"] == "ejb"
    m = model.lower()
    kw = dict(model=model, max_model_len=cfg.get("max_model_len", 4096 if ejb else 1365),
              tensor_parallel_size=1, trust_remote_code=True)
    if ejb:   # EditJudge-Bench runs: 512x512 JPEGs sent inline
        kw.update(gpu_memory_utilization=0.9, max_num_seqs=256, limit_mm_per_prompt={"image": 2, "video": 1})
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(jpeg_data_uri, {ex[k] for ex in examples for k in ("before", "after")}))
        url = jpeg_data_uri
    else:     # GenAI-Bench / EditReward-Bench runs: original files, resized by the model's processor
        kw.update(gpu_memory_utilization=0.95, limit_mm_per_prompt={"image": 2}, allowed_local_media_path="/")
        url = lambda p: f"file://{p}"  # noqa: E731
    if "qwen" in m and "llava" not in m:
        kw["mm_processor_kwargs"] = {"max_pixels": 512 * 512, "min_pixels": 256 * 256}
    if cfg.get("thinking"):
        kw["reasoning_config"] = {"reasoning_parser": "qwen3"}
    llm = LLM(**kw)

    if cfg.get("logprobs"):
        ids = [_token_id(model, "True"), _token_id(model, "False")]
        sp = SamplingParams(max_tokens=1, logprobs=2, temperature=0.0, allowed_token_ids=ids)
    elif cfg.get("thinking"):   # Qwen3 recommended sampling
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=cfg["max_tokens"],
                            thinking_token_budget=cfg["thinking_budget"])
    else:                       # greedy
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=cfg["max_tokens"])

    chat_kw = {"chat_template_kwargs": {"enable_thinking": bool(cfg.get("thinking"))}} if ejb else {}
    outputs = llm.chat([_messages(ex, url) for ex in examples], sampling_params=sp, **chat_kw)

    rows = []
    for out in outputs:
        o = out.outputs[0]
        if cfg.get("logprobs"):
            rows.append({"raw_output": o.text, "score": _p_true(o.logprobs, ids)})
            continue
        text = o.text
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        elif cfg.get("thinking") and not (getattr(o, "reasoning_content", None) or getattr(o, "thinking", None)):
            text = None   # stopped mid-reasoning: unparsed
        rows.append({"raw_output": o.text, "answer": text})
    return rows


def _p_true(logprobs, ids) -> float:
    """P(True) from the first token, renormalised over {True, False}."""
    if not logprobs or not logprobs[0]:
        return None
    lp = {i: v.logprob for i, v in logprobs[0].items()}
    t, f = lp.get(ids[0]), lp.get(ids[1])
    if t is None and f is None:
        return None
    if t is not None and f is not None:
        pt, pf = math.exp(t), math.exp(f)
        return pt / (pt + pf) if pt + pf > 0 else 0.5
    return min(1.0, math.exp(t)) if t is not None else 1.0 - min(1.0, math.exp(f))
