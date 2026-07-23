<div align="center">

# GEM-MM

### Entropy-Guided Multimodal Preference Alignment for Vision-Language Models

**Official code** for adapting [GEM](https://arxiv.org/abs/2511.13007) to vision-language models via fork-token entropy rewards and on-policy SEGA.

[![GitHub](https://img.shields.io/badge/GitHub-SNOWTEAM2023%2FGEM--MM-0B3D5C?style=for-the-badge&logo=github)](https://github.com/SNOWTEAM2023/GEM-MM)
[![License: MIT](https://img.shields.io/badge/License-MIT-2A9D8F?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Backbone-Qwen3--VL--4B-C1121F?style=for-the-badge)](https://huggingface.co/Qwen)

[Installation](#-installation) ·
[Quickstart](#-quickstart) ·
[Method](#-method) ·
[Results](#-results) ·
[Citation](#-citation)

<br/>

<img src="assets/poster.png" alt="GEM-MM conference poster" width="95%"/>

<sub>Conference poster · full size in <a href="assets/poster.png"><code>assets/poster.png</code></a></sub>

</div>

---

## Why GEM-MM?

Vision-language models perceive and reason well, but **preference alignment** in the multimodal setting is still brittle.  
**GEM-MM** brings GEM’s information-theoretic signal to VLMs:

| Idea | What we do |
|------|------------|
| **Fork entropy** | Reward candidates using entropy at uncertain “fork” tokens |
| **On-policy SEGA** | Sample $k$ responses, normalize advantages in-group, update the policy |
| **Vision-aware recipe** | Qwen3-VL messages, pixel bounds, `repo_limit128` prompting, **full FT** |

> Tagline: *Entropy for better alignment. Information for better answers.*

---

## News

- **2026-07** — Public code skeleton, configs, poster, and docs released in this repository.
- Paper draft & full table refresh in progress (SIMA / public benches).

---

## Highlights

- Extends **GEM → multimodal** (GEM-MM), not a text-only re-run  
- Fair bake-off under a **fixed 3000-sample** MM-RLHF budget  
- Strong baselines: **SFT / DPO / MM-DPO / mDPO / SIMA**  
- Reported GEM-MM runs use **full fine-tune** (+ grad checkpointing), not LoRA  
- Eval suite: preference / A/B / near-chosen + **MMHal · Hallusion · MME** (+ HAL)  
- Cross-dataset evidence on **RLAIF-V**

---

## Method

```text
  image + question  (x_I , x_Q)
            │
            ▼
   sample k on-policy candidates  { y_1 … y_k }
            │
            ▼
   fork-token entropy  +  final-answer term  →  reward R(y)
            │
            ▼
   group-normalized advantage A_k
            │
            ▼
   SEGA update  (full fine-tune of π_θ)
```

**Reward (sketch).** Keep the top-$m$ high-entropy fork positions $F$:

$$
R(y)=\lambda\cdot\frac{1}{|F|}\sum_{t\in F} H\!\big(\pi(\cdot\mid x,y_{<t})\big)+b_{\mathrm{final}}
$$

**Advantage.** Within each prompt group, standardize rewards → $A_k$, then apply the SEGA update.

Details, ablations, and VLM knobs: **[`docs/method.md`](docs/method.md)**

---

## Results

**Protocol.** Qwen3-VL-4B-Instruct · **3000** MM-RLHF train IDs · **1000** held-out eval · same prompt style for all methods.

| Method | Pref. Acc. (%) | Notes |
|--------|---------------:|-------|
| Base | 51.2 | Instruct checkpoint |
| SFT | 57.9 | |
| DPO | 58.9 | |
| MM-DPO | 58.1 | |
| mDPO | 58.4 | |
| SIMA | — | filling in (train done; MM-RLHF eval running) |
| **GEM-MM (ours)** | **61.7** | Full FT SEGA · primary paper row |

Near-chosen / public-bench / ablation tables ship with the paper draft.  
Reproduce the fair split: [`docs/reproduce.md`](docs/reproduce.md)

---

## Repository structure

```text
GEM-MM/
├── assets/              # poster & figures
├── configs/             # default + ablation YAMLs
├── docs/                # method · datasets · reproduce · checklist
├── examples/            # quickstart
├── scripts/             # train / eval entrypoints
├── src/gem_mm/          # library (config, entropy, SEGA, prompts)
└── tests/               # unit tests for reward helpers
```

---

## Installation

```bash
git clone https://github.com/SNOWTEAM2023/GEM-MM.git
cd GEM-MM

conda env create -f environment.yml
conda activate gem-mm

pip install -r requirements.txt
pip install -e .
```

**Hardware.** CUDA GPU recommended — ≥24 GB for 4B full FT; **40 GB** is comfortable with gradient checkpointing.

---

## Quickstart

```bash
# Paths (adjust to your machine)
export GEM_DATA_ROOT=/path/to/data
export GEM_MM_OUT_ROOT=/path/to/outputs

# Sanity-check config resolution
python scripts/train_gem_mm.py --config configs/default_gem_mm.yaml --dry-run

# Train GEM-MM (SEGA) — wire your local Qwen3-VL stack as documented
python scripts/train_gem_mm.py --config configs/default_gem_mm.yaml

# Preference eval on held-out pairs
python scripts/eval_preference.py --config configs/default_gem_mm.yaml \
  --checkpoint "$GEM_MM_OUT_ROOT/gem_mm/checkpoint-final"
```

More: [`examples/quickstart.md`](examples/quickstart.md) · configs in [`configs/`](configs/)

---

## Datasets & benchmarks

| Resource | Role |
|----------|------|
| [MM-RLHF](https://github.com/Kwai-Keye/MM-RLHF) | Main preference train / eval |
| [RLAIF-V](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) | Transfer / generalization |
| MMHal · Hallusion · MME | Public multimodal benches |
| POPE · AMBER · safety | API-free HAL suite |

Preparation notes: [`docs/datasets.md`](docs/datasets.md)

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/method.md`](docs/method.md) | Equations & VLM adaptations |
| [`docs/reproduce.md`](docs/reproduce.md) | Fair protocol & checklist |
| [`docs/datasets.md`](docs/datasets.md) | Data layout |
| [`docs/conference_repo_checklist.md`](docs/conference_repo_checklist.md) | What belongs in a submission repo |
| [`assets/poster.png`](assets/poster.png) | Printable landscape poster |

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{gemmm2026,
  title        = {GEM-MM: Entropy-Guided Multimodal Preference Alignment
                  for Vision-Language Models},
  author       = {Chua, Yuanshan and Zhao, Xuejiao},
  year         = {2026},
  howpublished = {\url{https://github.com/SNOWTEAM2023/GEM-MM}},
  note         = {Code and resources}
}
```

Please also cite the original text **GEM** paper when appropriate:  
[arXiv:2511.13007](https://arxiv.org/abs/2511.13007) · machine-readable: [`CITATION.cff`](CITATION.cff)

---

## Acknowledgments

We build on GEM, DPO-family preference learning (including multimodal variants such as MM-DPO, mDPO, and SIMA), and the Qwen3-VL model family. Experiments use open benchmarks (MMHal-Bench, HallusionBench, MME, POPE, AMBER).

---

## License

Released under the [MIT License](LICENSE).

---

## Contact

| | |
|--|--|
| **Yuanshan Chua** | [`YCHUA060@e.ntu.edu.sg`](mailto:YCHUA060@e.ntu.edu.sg) |
| **Xuejiao Zhao** | [`xjzhao@ntu.edu.sg`](mailto:xjzhao@ntu.edu.sg) |
| **Affiliation** | Nanyang Technological University |
| **Org** | [SNOWTEAM2023](https://github.com/SNOWTEAM2023) |

<div align="center">
<sub>Maintained at <a href="https://github.com/SNOWTEAM2023/GEM-MM">github.com/SNOWTEAM2023/GEM-MM</a></sub>
</div>
