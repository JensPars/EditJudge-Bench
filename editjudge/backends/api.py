"""Proprietary judges: OpenAI, Anthropic and Gemini APIs.

Calls run in 16 threads and are checkpointed to a JSONL file next to the output,
so an interrupted run resumes where it stopped.
"""
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

RETRY_WAIT = [2, 5, 10, 30]
TRANSIENT = ("429", "rate", "quota", "500", "503", "529", "overloaded", "timeout", "connection")


def _image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > 512 * 512:
        s = (512 * 512 / (w * h)) ** 0.5
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def _b64_jpeg(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _retry(call):
    for wait in RETRY_WAIT + [0]:
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            if wait and any(k in str(e).lower() for k in TRANSIENT):
                time.sleep(wait)
                continue
            raise


# ---------------------------------------------------------------------------
# Providers: each returns call(example) -> text
# ---------------------------------------------------------------------------

def _openai(cfg):
    from openai import OpenAI
    client = OpenAI()

    def call(ex):
        content = [{"type": "text", "text": v} if k == "text" else
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64_jpeg(_image(ex[v]))}}
                   for k, v in ex["parts"]]
        kw = dict(model=cfg["model"], max_completion_tokens=cfg["max_tokens"],
                  messages=([{"role": "system", "content": ex["system"]}] if ex["system"] else [])
                  + [{"role": "user", "content": content}])
        if cfg.get("reasoning_effort"):
            kw["reasoning_effort"] = cfg["reasoning_effort"]
        r = client.chat.completions.create(**kw)
        return (r.choices[0].message.content or "") if r.choices else ""
    return call


def _anthropic(cfg):
    import anthropic
    client = anthropic.Anthropic()

    def call(ex):
        content = [{"type": "text", "text": v} if k == "text" else
                   {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": _b64_jpeg(_image(ex[v]))}}
                   for k, v in ex["parts"]]
        kw = dict(model=cfg["model"], max_tokens=cfg["max_tokens"], messages=[{"role": "user", "content": content}])
        if cfg.get("thinking_effort"):
            kw.update(thinking={"type": "adaptive"}, output_config={"effort": cfg["thinking_effort"]})
        else:
            kw["thinking"] = {"type": "disabled"}
        if ex["system"]:
            kw["system"] = ex["system"]
        r = client.messages.create(**kw)
        return "".join(b.text for b in r.content if b.type != "thinking" and hasattr(b, "text"))
    return call


def _gemini(cfg):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    thinking = cfg.get("thinking", False)

    def call(ex):
        contents = [v if k == "text" else _image(ex[v]) for k, v in ex["parts"]]
        conf = types.GenerateContentConfig(
            system_instruction=ex["system"] or None, temperature=0.0,
            # thinking tokens count against the output budget
            max_output_tokens=cfg["max_tokens"] + (8192 if thinking else 0),
            thinking_config=None if thinking else types.ThinkingConfig(thinking_budget=0))
        r = client.models.generate_content(model=cfg["model"], contents=contents, config=conf)
        parts = r.candidates[0].content.parts if r.candidates and r.candidates[0].content else None
        return "".join(p.text for p in parts or [] if p.text is not None and not p.thought)
    return call


PROVIDERS = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini}


def run(cfg: dict, examples: list, checkpoint: Path) -> list:
    call = PROVIDERS[cfg["backend"]](cfg)
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            e = json.loads(line)
            done[e["idx"]] = e["output"]
        print(f"  Resuming: {len(done)}/{len(examples)} done")
    lock = threading.Lock()
    errors = []

    def work(i):
        try:
            out = _retry(lambda: call(examples[i]))
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
            return
        with lock, open(checkpoint, "a") as f:
            f.write(json.dumps({"idx": i, "output": out}) + "\n")
            done[i] = out
        if cfg["backend"] == "gemini":
            time.sleep(0.1)

    todo = [i for i in range(len(examples)) if i not in done]
    with ThreadPoolExecutor(16) as pool:
        list(tqdm(pool.map(work, todo), total=len(todo), desc=cfg["backend"]))
    if errors:
        raise SystemExit(f"{len(errors)} API calls failed (first: {errors[0][:200]}). Re-run to retry them.")
    return [{"raw_output": done[i], "answer": done[i]} for i in range(len(examples))]
