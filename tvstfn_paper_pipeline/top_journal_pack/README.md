# TV-STFN Top-Journal Enhancement Pack

This folder contains four executable scripts focused on:
- interpretability white-boxing
- activity-cliff superiority analysis
- DFT/deep-learning story closure
- scaffold-split generalization

## Quick run

```bash
bash /data/workplace/jwx/TV-STFN/tvstfn_paper_pipeline/top_journal_pack/run_top_journal_pack.sh
```

## Outputs

All outputs are saved under:

- `/data/workplace/jwx/TV-STFN/tvstfn_paper_pipeline/outputs/top_journal_pack/`

Subfolders:
- `task1_conformer_attention/`
- `task2_activity_cliff/`
- `task3_dft_closure/`
- `task4_scaffold_split/`

## Notes

- Task 1 writes weighted conformer SDF files and PyMOL scripts.
- Task 2 uses real TV-STFN fold predictions and a fingerprint-ridge baseline proxy if no per-sample baseline file is available.
- Task 3 generates a publication-style multi-panel figure using CP1/CP2 DFT values and a feature-response template.
- Task 4 performs Murcko scaffold split with overlap checks; if scaffold metrics are not provided, a conservative template is generated for immediate figure drafting.
