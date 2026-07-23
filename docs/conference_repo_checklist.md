# What usually goes in a conference-code GitHub repo

Use this as a living checklist for AAAI / NeurIPS / CVPR-style releases.

## Must-have

- [x] Clear **README** (abstract, install, quickstart, results teaser, citation)  
- [x] **LICENSE** (MIT here)  
- [x] **Citation** (`CITATION.cff` + BibTeX in README)  
- [x] Reproducible **environment** (`requirements.txt` / `environment.yml`)  
- [x] **Configs** for main + ablations  
- [x] Minimal **runnable entrypoints** (`scripts/`)  
- [x] Method / data / reproduce **docs**  
- [x] `.gitignore` that keeps checkpoints & raw data out  

## Strongly recommended

- [x] Poster / teaser figure in `assets/`  
- [x] Conference repo checklist (this file)  
- [ ] Model / dataset cards or HF links once public  
- [ ] Exact command lines that reproduce each table row  
- [ ] Seed + hardware notes  
- [ ] Expected runtime / GPU hours  

## Optional but polished

- [x] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`  
- [ ] CI lint (`ruff`) on PRs  
- [ ] Demo Colab / Gradio  
- [ ] Video / slides  

## Do **not** put in the public repo

- API keys, wandb keys, private emails beyond contact  
- Full training checkpoints (host on HF / Zenodo)  
- Raw image corpora  
- Internal experiment trackers with absolute lab paths  
- Fabricated historical git timelines (prefer honest progressive commits)
