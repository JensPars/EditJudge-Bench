"""Prompt templates, prompt layout and output parsers.

Template text is verbatim from the paper runs. A prompt is laid out as a list of
parts: ("text", str) or ("image", "before" | "after"). Each backend turns the parts
into its own request format, so every backend sees the same layout.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from editjudge import kris_bench

BEFORE, AFTER = "<<BEFORE_IMAGE>>", "<<AFTER_IMAGE>>"
_MARKERS = re.compile(f"({re.escape(BEFORE)}|{re.escape(AFTER)})")


@dataclass(frozen=True)
class Template:
    system: str
    user: str              # placeholders: {instruction}, {question}
    parser: str            # vie | imgedit | kris | binary | none
    yesno: bool = False    # fill {question} with a yes/no question
    captions: Optional[Tuple[str, str]] = None  # text part before each image (KRIS layout)


_BINARY_SYSTEM = "You are an expert at evaluating image edits. Answer with only 'True' or 'False'."
_BINARY_ANSWER = 'Answer with only "True" or "False".'

TEMPLATES = {
    # VIEScore, semantic-consistency prompt (two 0-10 scores)
    "vie_tie_0shot": Template(
        system="""You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}""",
        user="""RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.

From scale 0 to 10:
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: {instruction}""",
        parser="vie",
    ),
    # KRIS-Bench instruction-following prompt (1-5)
    "kris_instruction": Template(
        system="",
        user=kris_bench.prompt_instruction_following.replace("{instruct}", "{instruction}"),
        parser="kris",
        captions=(kris_bench.ORIGINAL_IMAGE_CAPTION, kris_bench.EDITED_IMAGE_CAPTION),
    ),
    # ImgEdit, zero-shot (three 1-5 scores)
    "imgedit": Template(
        system="You are an expert evaluator of image edits.",
        user="""You are given an original image and an edited image. The editing instruction is: "{instruction}"

Please evaluate the edited image based on the following criteria:

1. Prompt Compliance (1-5): Does the edit follow the instruction?
   - 5: Perfectly follows instruction
   - 4: Mostly follows instruction with minor deviations
   - 3: Partially follows instruction
   - 2: Barely follows instruction
   - 1: Does not follow instruction

2. Visual Seamlessness (1-5): How natural and seamless is the edit?
   - 5: Perfectly seamless, no visible artifacts
   - 4: Mostly seamless with minor imperfections
   - 3: Noticeable but acceptable seams
   - 2: Obvious visual artifacts
   - 1: Very poor visual quality

3. Physical & Detail Fidelity (1-5): Are physical properties and details preserved?
   - 5: Perfect physical realism and detail preservation
   - 4: Good realism with minor detail loss
   - 3: Acceptable but some unrealistic elements
   - 2: Noticeable physical inconsistencies
   - 1: Severely unrealistic

Format your response as:
Prompt Compliance: [score]
Visual Seamlessness: [score]
Physical & Detail Fidelity: [score]""",
        parser="imgedit",
    ),
    # ImgEdit-Judge's own grading prompt
    "imgedit_judge": Template(
        system="",
        user="""
You are a data rater specializing in grading attribute alteration edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the attribute change on a 5-point scale from three perspectives:

Prompt Compliance
1  Target not adjusted, wrong object touched, or geometry changed.
2  Right object but wrong attribute value/direction; only part edited; other objects also altered; slight stretch/crop.
3  Mainly correct object and attribute, yet large hue/brightness/texture error; minor collateral edits; visible jaggies/distortion.
4  All requested objects adjusted, only their attributes changed; shape kept; small inaccuracy in colour, material or amount.
5  Exactly and only the requested objects adjusted; colour, material, gloss etc. match the prompt perfectly; shape 100% intact; zero unintended edits.

Visual Seamlessness
1  Massive colour spill, mosaics or heavy noise; image nearly unusable.
2  Clear smears/bleeding on edges; abrupt resolution or tone shift; highlights/shadows clipped; background gaps.
3  Overall palette OK but local tone or grain conflicts; soft edges; noticeable disharmony.
4  Style unified, transitions smooth; only slight edge artefacts visible when zoomed.
5  No detectable edit traces; colours/materials fuse with scene lighting; edit area practically invisible.

Physical & Detail Fidelity
1  Object floating, interpenetrating, or severe perspective/light mismatch; background badly warped.
2  Missing shadows/highlights; wrong reflection direction; background visibly discoloured or distorted.
3  Light, perspective and contact surface largely correct; minor acceptable flaws; background only locally affected.
4  Adjusted material interacts believably with scene; shadows, highlights, reflections handled well; original details preserved.
5  High physical realism: fine micro-highlights, diffuse bounce, subsurface effects present; overall scene realism improved.
The second and third score should no higher than first score!!!

Example Response Format:
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Prompt Compliance: A number from 1 to 5.
Visual Seamlessness: A number from 1 to 5.
Physical & Detail Fidelity: A number from 1 to 5.
editing instruction is : {instruction}.

Below are the images before and after editing:
""",
        parser="imgedit",
    ),
    # Binary log-prob baseline (Table 1): yes/no question before the answer line
    "binary_q_images_last": Template(
        system=_BINARY_SYSTEM,
        user=f"""Edit instruction: "{{instruction}}"

Before Image:
{BEFORE}
After Image:
{AFTER}
{{question}}

{_BINARY_ANSWER}""",
        parser="binary",
        yesno=True,
    ),
    # Same prompt without the question (Table 3, GenAI-Bench / EditReward-Bench)
    "binary_tagged_images_last": Template(
        system=_BINARY_SYSTEM,
        user=f"""Edit instruction: "{{instruction}}"

Before Image:
{BEFORE}
After Image:
{AFTER}
{_BINARY_ANSWER}""",
        parser="binary",
    ),
    # Earlier binary prompt (Table 3, Qwen2.5-VL-7B log-prob row)
    "binary_simple": Template(
        system=_BINARY_SYSTEM,
        user='Given the edit instruction: "{instruction}" Look at the first before image and the second '
             'after image carefully. Answer with only "True" or "False".',
        parser="binary",
    ),
    # Reward models (EditReward, EditScore) get the raw instruction
    "editreward": Template(system="", user="{instruction}", parser="none"),
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_prompt(template: str, instruction: str, image_position: Optional[str]) -> Tuple[str, List[tuple]]:
    """(system prompt, user parts) for one instruction.

    Templates with image markers place the images at the markers. Otherwise
    ``image_position`` decides: "before" = images first, "after" = text first.
    """
    tpl = TEMPLATES[template]
    question = instruction_to_question(instruction) if tpl.yesno else instruction
    text = tpl.user.format(instruction=instruction, question=question)

    if BEFORE in text or AFTER in text:
        parts = []
        for seg in _MARKERS.split(text):
            if seg == BEFORE:
                parts.append(("image", "before"))
            elif seg == AFTER:
                parts.append(("image", "after"))
            elif seg.strip():
                parts.append(("text", seg.strip()))
        return tpl.system, parts

    if image_position not in ("before", "after"):
        raise ValueError(f"template {template!r} needs image_position = 'before' or 'after'")
    images = []
    for caption, which in zip(tpl.captions or (None, None), ("before", "after")):
        if caption:
            images.append(("text", caption))
        images.append(("image", which))
    return tpl.system, ([("text", text)] + images if image_position == "after" else images + [("text", text)])


# ---------------------------------------------------------------------------
# Yes/no question for the binary_q template
# ---------------------------------------------------------------------------

_PAST = {"move": "moved", "make": "made", "change": "changed", "rotate": "rotated", "zoom": "zoomed",
         "remove": "removed", "add": "added", "close": "closed", "open": "opened",
         "stretch": "stretched", "compress": "compressed"}
_NO_EDIT = ("do not", "leave the", "the scene", "no edits", "keep the")
_SPLIT = re.compile(r"\s(to|from|in|out|by|\d+|clockwise|counterclockwise|counter-clockwise|horizontally|vertically|"
                    r"left|right|up|down|upleft|upright|downleft|downright|forward|backward|towards|away)\b")


def instruction_to_question(instruction: str) -> str:
    """"Move the oven right." -> "Has the oven been moved right?" """
    text = instruction.strip().rstrip(".")
    if text.lower().startswith(_NO_EDIT):
        return "Has the scene been left unchanged?"
    verb, _, rest = text.partition(" ")
    pp = _PAST.get(verb.lower())
    if pp is None:
        raise ValueError(f"no question rule for instruction: {instruction!r}")
    if verb.lower() == "make":
        obj, _, adj = rest.rpartition(" ")
        if adj in ("length", "width", "height", "depth"):  # "shorter in length"
            obj, _, head = obj.rpartition(" ")
            obj, _, cmp_ = obj.rpartition(" ")
            adj = f"{cmp_} {head} {adj}"
        return f"Has {obj} been made {adj}?"
    m = _SPLIT.search(rest)
    obj, comp = (rest[:m.start()], rest[m.start():]) if m else (rest, "")
    return f"Has {obj} been {pp}{comp}?"


# ---------------------------------------------------------------------------
# Parsers: raw output -> score in [0, 1] (None = unparsed)
# ---------------------------------------------------------------------------

def _vie(text: str) -> Optional[float]:
    m = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 20
    s1 = re.search(r'score1\s*[=:]\s*(\d+(?:\.\d+)?)', text, re.I)
    s2 = re.search(r'score2\s*[=:]\s*(\d+(?:\.\d+)?)', text, re.I)
    if s1 and s2:
        return (float(s1.group(1)) + float(s2.group(1))) / 20
    nums = [float(n) for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', text) if 0 <= float(n) <= 10]
    return (nums[-2] + nums[-1]) / 20 if len(nums) >= 2 else None


def _imgedit(text: str) -> Optional[float]:
    pats = [r'Prompt Compliance[:\s]*(\d+)', r'Visual Seamlessness[:\s]*(\d+)',
            r'Physical\s*(?:&|and)?\s*Detail Fidelity[:\s]*(\d+)']
    vals = [float(m.group(1)) for p in pats if (m := re.search(p, text, re.I))]
    return sum(vals) / len(vals) / 5 if vals else None


def _kris(text: str) -> Optional[float]:
    # JSON object first, then "instruction_score": N (also in truncated JSON), N/5, N out of 5
    m = re.search(r"\{[^{}]*instruction_score[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0)).get("instruction_score")
            if v is not None:
                return float(int(v)) / 5
        except (ValueError, TypeError, AttributeError):
            pass
    for p in (r'instruction[_\s]*score"?\s*[:：]?\s*([1-5])', r"([1-5])\s*/\s*5", r"([1-5])\s+out\s+of\s+5"):
        if m := re.search(p, text, re.I | re.DOTALL):
            return float(m.group(1)) / 5
    return None


def _binary(text: str) -> Optional[float]:
    t = text.strip().lower()
    i, j = t.rfind("true"), t.rfind("false")
    if i != j:
        return 1.0 if i > j else 0.0
    if re.search(r'\byes\b', t):
        return 1.0
    if re.search(r'\bno\b', t):
        return 0.0
    return None


PARSERS = {"vie": _vie, "imgedit": _imgedit, "kris": _kris, "binary": _binary}


def parse(template: str, text: Optional[str]) -> Optional[float]:
    return None if text is None else PARSERS[TEMPLATES[template].parser](text)
