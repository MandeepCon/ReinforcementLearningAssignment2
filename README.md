# CSCN 8020 — Assignment 2: Q-Learning on Taxi-v3

> **Research Question → Results**  
> **Can tabular Q-Learning learn an efficient Taxi-v3 policy, and which α/ε works best?**  
> **Yes.** Training converges with rising returns and falling steps/episode. The best configuration is auto-selected by **highest last-100 avg return**, re-trained, and exported as PNG plots (no `.npy` Q-table saved).

---

## Overview

This repo trains a **tabular Q-Learning** agent on **Gymnasium’s Taxi-v3** (500 states, 6 actions) and performs **hyperparameter sweeps** over α (learning rate) and ε (exploration). It exports:
- Per-episode **learning curves** (returns + rolling mean; steps/episode)
- A compact **Results Card** PNG for the **best run**
- A **plots-only PDF report** consolidating figures and the summary table

> All artifacts are written under `artifacts/rl_taxi/run_<timestamp>/`.

---

## Repo Structure

```
.
├─ notebooks/ or root .ipynb         # Your Jupyter notebook
├─ assignment2_utils.py              # (Course helper, optional)
├─ requirements.txt                  # Project dependencies
├─ artifacts/
│  └─ rl_taxi/
│     └─ run_<timestamp>/
│        ├─ results_png/             # Best-run PNGs (curves, results card)
│        ├─ summary_runs.csv         # All runs summary (baseline + sweeps)
│        ├─ summary_sorted.csv       # Sorted (best on top)
│        └─ assignment2_report_plots_only.pdf
└─ README.md                         # This file
```

---

## Environment & Requirements

- **Python:** 3.10–3.12 recommended  
- **Key packages:** `gymnasium`, `numpy`, `pandas`, `matplotlib`, `tqdm`  
- For **human rendering** demos, install `pygame`.

Install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

If you need to generate your own:
```bash
pip freeze > requirements.txt
# PowerShell:
# pip freeze | Out-File -Encoding utf8 requirements.txt
```

---

## Quick Start (Notebook)

Open the notebook and run cells in order:

1) **Setup & Imports**  
   - Seeds, plotting defaults, and **Gymnasium→gym alias** for helper compatibility:
   ```python
   import gymnasium as gym, sys
   sys.modules.setdefault("gym", gym)  # so assignment2_utils that imports 'gym' still works
   ```

2) **Make Env & Sanity Check**
   ```python
   env = gym.make("Taxi-v3")
   obs, _ = env.reset(seed=73)
   print(env.observation_space.n, env.action_space.n)  # 500, 6
   ```

3) **Hyperparameters**  
   Baseline `α=0.10, ε=0.10, γ=0.90`, sweeps over α and ε independently.

4) **Train (Q-Learning)**  
   - Runs baseline + α-sweep + ε-sweep  
   - Logs per-episode **return** and **steps**

5) **Summarize & Select Best**  
   - Builds `summary_sorted` and picks **best** by last-100 avg return

6) **Confirm Best & Export PNGs**  
   - Re-trains best config  
   - Saves **learning curves** + **Results Card** to `results_png/`

7) **Generate Plots-Only PDF**  
   - `assignment2_report_plots_only.pdf` under the run folder

---

## Outputs

- **PDF Report:**  
  `artifacts/rl_taxi/run_<timestamp>/assignment2_report_plots_only.pdf`

- **Best-Run PNGs (folder):**  
  `artifacts/rl_taxi/run_<timestamp>/results_png/`  
  - `<tag>_confirm_curves.png` — returns (with rolling mean) + steps/episode  
  - `<tag>_confirm_results_card.png` — α/ε/γ, episodes, Avg Return, Last-100, Avg Steps/Ep, Total Steps, Elapsed

- **CSV Summaries:**  
  - `summary_runs.csv` (all runs)  
  - `summary_sorted.csv` (best on top)

---

## Report Snippets (Copy-Paste)

**Results Box (auto-filled in notebook):**
```markdown
**Results → <best tag>**  
α=<alpha>, ε=<epsilon>, γ=<gamma>  
Episodes=5000 • Avg Return=<…> • Last-100=<…>  
Avg Steps/Ep=<…> • Total Steps=<…> • Elapsed=<…>s
```

**Citation (Gymnasium Taxi):**  
> Farama Foundation. *Gymnasium – Taxi-v3 (toy_text)*. https://gymnasium.farama.org/environments/toy_text/taxi/

---

## Notes & Tips

- The project **does not save** the Q-table (`.npy`) by design; only plots and CSVs are exported.  
- To keep the **course helper** working, we map **Gymnasium** to `gym` at import time.  
- If your instructor requires both **Q-Learning and DQN**, add a compact DQN section (MLP, replay, target network) and compare plots.

---

## Academic Integrity

All training, evaluation, logging, and plotting code is **originally structured** (naming, APIs, and file layout differ from peers). The helper (if used) is restricted to environment description only.

---

## License

Use for educational purposes within the course. Adapt/extend responsibly.
