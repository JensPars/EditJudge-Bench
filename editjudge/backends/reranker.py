"""Qwen3-VL-Reranker: P(yes) from a head built from lm_head[yes] - lm_head[no]."""
import gc

from PIL import Image
from tqdm import tqdm

SYSTEM = ("Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
          "Note that the answer can only be \"yes\" or \"no\".")
INSTRUCT = "Determine whether the edited image correctly follows the editing instruction applied to the original image."


def run(cfg: dict, examples: list) -> list:
    import torch
    import torch.nn.functional as F
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # Image sizes as in the paper runs: EditJudge-Bench 256^2-512^2 px, GAB/ERB 128^2-256^2 px
    ejb = cfg["benchmark"] == "ejb"
    lo, hi = (256 * 256, 512 * 512) if ejb else (128 * 128, 256 * 256)

    lm = Qwen3VLForConditionalGeneration.from_pretrained(cfg["model"], dtype=torch.bfloat16,
                                                         trust_remote_code=True).cuda()
    base = lm.model.eval()
    # PyTorch 2.9 Conv3D regression: run the patch embedding as a linear layer
    conv = lm.visual.patch_embed.proj
    n_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1] * conv.kernel_size[2]
    w, b = conv.weight.data.view(conv.out_channels, n_in), conv.bias.data if conv.bias is not None else None
    lm.visual.patch_embed.forward = lambda x: F.linear(x.reshape(x.shape[0], n_in), w, b)

    proc_kw = dict(trust_remote_code=True, padding_side="left")
    if ejb:
        proc_kw.update(min_pixels=lo, max_pixels=hi)
    proc = AutoProcessor.from_pretrained(cfg["model"], **proc_kw)
    vocab = proc.tokenizer.get_vocab()
    head = torch.nn.Linear(lm.lm_head.weight.size(1), 1, bias=False)
    with torch.no_grad():
        head.weight[0] = lm.lm_head.weight.data[vocab["yes"]] - lm.lm_head.weight.data[vocab["no"]]
    head = head.to(base.dtype).cuda().eval()
    del lm.lm_head
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for ex in tqdm(examples, desc="reranker"):
        img = lambda k: {"type": "image", "image": Image.open(ex[k]).convert("RGB"),  # noqa: E731
                         "min_pixels": lo, "max_pixels": hi}
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": f"<Instruct>: {INSTRUCT}<Query>:"}, img("before"),
                                             {"type": "text", "text": ex["instruction"]},
                                             {"type": "text", "text": "\n<Document>:"}, img("after")]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, _, _ = process_vision_info(msgs, return_video_kwargs=True, return_video_metadata=True)
        inputs = proc(text=[text], images=images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            p = torch.sigmoid(head(base(**inputs).last_hidden_state[:, -1])).item()
        rows.append({"raw_output": f"P(yes)={p:.6f}", "score": float(p)})
    return rows
