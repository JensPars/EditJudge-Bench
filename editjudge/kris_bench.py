"""KRIS-Bench prompts, copied byte-exact from
https://github.com/mercurystraw/Kris_Bench (utils/prompts.py @ cf14752, Apache-2.0,
see third_party/kris_bench/). Do not reformat this file.
"""

prompt_consist = """
You are a professional digital artist and image evaluation specialist.

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate the visual consistency between the original and edited images, focusing exclusively on elements that are NOT specified for change in the instruction**. That is, you should only consider whether all non-instructed details remain unchanged. Do **not** penalize or reward any changes that are explicitly required by the instruction.

## Evaluation Scale (1 to 5):
You will assign a **consistency_score** according to the following rules:
- **5 Perfect Consistency**: All non-instruction elements are completely unchanged and visually identical.
- **4 Minor Inconsistency**: Only one very small, non-instruction detail is different (e.g., a tiny accessory, a subtle shadow, or a minor background artifact).
- **3 Noticeable Inconsistency**: One clear non-instruction element is changed (e.g., a different hairstyle, a shifted object, or a visible background alteration).
- **2 Significant Inconsistency**: Two or more non-instruction elements have been noticeably altered.
- **1 Severe Inconsistency**: Most or all major non-instruction details are different (e.g., changed identity, gender, or overall scene layout).

## Guidance:
- First, **identify all elements that the instruction explicitly allows or requires to be changed**. Exclude these from your consistency check.
- For all other elements (e.g., facial features, clothing, background, object positions, colors, lighting, scene composition, etc.), **compare Image B to Image A** and check if they remain visually identical.
- If you observe any change in a non-instruction element, note it and consider its impact on the score.
- If the instruction is vague or ambiguous, make a best-effort factual inference about which elements are intended to change, and treat all others as non-instruction elements.

## Note:
- **Do not penalize changes that are required by the instruction.**
- **Do not reward or penalize the quality or correctness of the instructed change itself** (that is evaluated separately).
- If the edited image introduces new artifacts, objects, or changes to non-instruction elements, this should lower the consistency score.

## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
First, clearly explain your comparison process: list each major non-instruction element and state whether it is consistent (unchanged) or inconsistent (changed), with brief reasoning.
Then, provide your evaluation in the following JSON format:
{{
"reasoning": **Compared to original image**, [list of non-instruction elements that changed or remained the same] **in the edited image**. 
"consistency_score": X
}}
"""

prompt_instruction_following = """
You are a professional digital artist and image evaluation specialist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules. 

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the presence and correctness of the specified changes**. 

You must:
**Identify detailed visual differences** between Image A and Image B **correctly and faithfully**.
Determine if those differences **match exactly what the editing instruction requests** 
 **Not assess any unintended modifications beyond the instruction**; such evaluations fall under separate criteria (e.g., visual consistency).
**Be careful**, an edit may introduce visual change without fulfilling the actual instruction (e.g., replacing the object instead of modifying it)

## Reasoning:
You must follow these reasoning steps before scoring:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position) In this step, you don't have to use information from the editing instruction.
**2. Expected Visual Caption**: Write a factual description of how the edited image should look if the instruction were perfectly followed.
**3. Instruction Match**: 
Compare the observed differences in **1** to the expected change in **2**:
- Was the correct object modified (not replaced)?
- Was the requested attribute (e.g., size, color, position) modified as intended?
- Is the degree of modification accurate (e.g., “match size,” “slightly increase,” etc.)?
**4. Decision**: Use the 1–5 scale to assign a final score.

## Evaluation Scale (1 to 5):
You will assign an **instruction_score** with following rule:
- **5 Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present and accurate. 
- **4 Minor Omission**: The core change is made, but **minor detail** is missing or slightly incorrect. 
- **3 Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
- **2 Major Omission**: Most of the required changes are missing or poorly implemented. 
- **1 Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 

Example: 
Instruction: Adjust the size of the apple to match the size of the watermelon
{{
  "instruction_score": 3,
  "reasoning": "1. Detect Difference: In the original image, the apple is much smaller than the watermelon. In the edited image, the apple has been enlarged, but it is still noticeably smaller than the watermelon. 2. Expected Visual Caption: The apple should be resized so that it visually matches the watermelon in size—approximately the same height and overall volume. 3. Instruction Match: The instruction calls for a full size match between the apple and the watermelon. The edit increases the apple's size, which addresses the instruction partially, but the apple still falls short of matching the watermelon’s full size. The core concept is attempted, but not fully realized. 4. Decision: Because the size change was made but not to the full extent required, this counts as 3 partial compliance."
}}

## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
Look at the input again, provide the evaluation score and the explanation in the following JSON format:
{{
"instruction_score": X,
"reasoning": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision
}}
"""

# KRIS-Bench (metrics_common.evaluate_with_gpt) sends one user message, no
# system prompt: the prompt text, then each image preceded by its own text part.
ORIGINAL_IMAGE_CAPTION = "This is the original image:"
EDITED_IMAGE_CAPTION = "This is the edited image:"
