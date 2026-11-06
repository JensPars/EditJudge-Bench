# EditJudge-Bench

**Auditing VLM Image-Edit Judges with Synthetic Ground Truth**
Jens Parslov, Marco Schouten, Dim P. Papadopoulos · NeurIPS 2026, Track on Evaluations and Datasets

[Project page](https://jenspars.github.io/EditJudge-Bench/) · [Paper](https://openreview.net/forum?id=eynW4BxY28) · [Dataset](https://huggingface.co/datasets/JensParslov/EditJudge-Bench)

![EditJudge-Bench overview: a 3D scene is edited and rendered as a before/after pair, and a VLM judge scores it against an instruction](docs/assets/teaser.png)

<sub><b>Auditing VLM image-edit judges.</b> We procedurally alter 3D scenes and render controlled before/after image pairs spanning nine edit types, then use these pairs to audit VLM image-edit judges, exposing systematic reliability gaps in current VLM-as-a-judge protocols.</sub>

Vision-language models are often used as judges for image-editing benchmarks, but
can they actually see the edits they score? EditJudge-Bench tests this with 1,382
before/after image pairs rendered in Blender. Every edit is a known change to a 3D
scene, so the ground truth is exact. A judge is shown an image pair with either the
correct instruction or a wrong one, and is scored on telling them apart.

## Code

**Coming soon.** This repository will hold the code to run judges on the benchmark,
the saved predictions behind every number in the paper, and the scripts that
rebuild the paper's tables and figures.

## Citation

```bibtex
@inproceedings{parslov2026editjudge,
  title     = {EditJudge-Bench: Auditing VLM Image-Edit Judges with Synthetic Ground Truth},
  author    = {Parslov, Jens and Schouten, Marco and Papadopoulos, Dim P.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Track on Evaluations and Datasets},
  year      = {2026}
}
```

## License

The dataset is released under CC-BY-NC-4.0.
