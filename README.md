<p align="center">
  <img src="NeurInSpectre2.png" alt="NeurInSpectre Logo" width="600"/>
</p>

# NeurInSpectre

A neural-network security-analysis framework for offensive and defensive
AI security operations, supporting two CCS 2026 submissions that share
this codebase but evaluate disjoint claims.

This top-level README is an index. **Reviewers should follow the
paper-specific README for the submission they are reviewing**:

| If you are reviewing… | Follow… | One-command reproduction |
|---|---|---|
| **The offensive paper**<br/>*"NeurInSpectre: An Offensive Framework for Breaking Gradient Obfuscation in AI Safety Systems via Spectral-Volterra-Krylov Analysis"* | [`README_OFFSEC.md`](README_OFFSEC.md) → [`AE.md`](AE.md) | `bash scripts/reproduce_table8.sh` |
| **The detection paper**<br/>*"NeurInSpectre: A Three-Layer Mathematical Framework for Gradient Obfuscation Detection in Adversarial Machine Learning"* | [`README_DETECTION.md`](README_DETECTION.md) → [`QUICKSTART_CCS.md`](QUICKSTART_CCS.md) | `bash scripts/reproduce_detection.sh` |

The two reproduction paths are **fully independent** — outputs are
written to non-overlapping subdirectories (`results/table8_run_v2/` +
`results/table5_rigor_production/` for the offensive paper;
`results/detection/` for the detection paper). You only need to run the
quickstart for the paper you are reviewing.

---

## Repository at a glance

- **Code:** `neurinspectre/` (~77 000 non-blank lines of GPU-accelerated
  Python). Single Python package shared by both papers; each paper
  exercises a different subset.
- **Scripts:** `scripts/reproduce_*.sh` (per-paper harnesses) +
  per-module CLIs.
- **Models:** `models/` (TorchScript artifacts pinned by SHA-256 in
  `*.meta.json` sidecars). The detection paper additionally pulls the
  Carmon2019Unlabeled checkpoint via `scripts/download_carmon2019.py`.
- **As-run artifacts:** `results/` (single-seed outputs that back the
  paper tables; the offensive paper's main matrix is
  `results/table8_run_v2/`, the detection paper's is
  `results/detection/`).
- **Audit / claim trace:** `DRAFT_CODE_AUDIT.md` (exhaustive
  claim → code → command → artifact mapping for the offensive paper).

---

## Install (5 min, once for either paper)

```bash
python3.10 -m venv .venv-neurinspectre
source .venv-neurinspectre/bin/activate
pip install -U pip
pip install -e ".[dev]"
pip install -r requirements-frozen.txt
neurinspectre doctor
```

After install, jump to whichever paper-specific README applies (table at top).

---

## Smoke tests (fast, runs without paper-grade compute)

```bash
# Offensive paper smoke test (~5 minutes; exercises the full Table 8 pipeline)
neurinspectre table2-smoke --output-dir results/smoke

# Detection paper smoke test (~5 seconds; Tables 1–4 synthetic only)
python scripts/run_synthetic_experiments.py --output-dir results/smoke
```

---

## License

MIT (see `LICENSE`).

---

## Contact

For double-blind review questions, please use the OpenReview / HotCRP
submission system. Post-acceptance contact details will be added here.
