# Report bf16 — reproduction matrice MI300A en bf16 (dev_hpe branch)

Référentiel fp32 = `report_mono_node_fp32.html` (snapshot pre-bf16 switch).
Source CSV partagé fp32+bf16 : `results.csv` (colonne `precision`).

Convention : **5 lignes par run** — setup / perf / observation / vs fp32 / verdict.

---

## Run 1 — Job 5078636 : 1n × 4 APUs b40 bf16

- **Setup** : `RUN_TAG=b40 BIND_STRATEGY=mi300_srun4 ./launch.parsable.sh 1 4 MI300 48` sur a1004. Premier test bf16 (dev trainer.py force `torch.bfloat16` partout : student/teacher backbones, heads, losses).
- **Perf** : médiane warm steps 100-250 = **0.50 it/s = 80.0 img/s**. ETA 10M = 34.7h (1.45 j). Cancel à step 250 (suffisant pour warm reading, on saved 4 mesures stables).
- **Observation loss** : trajectoire NON monotone — pic à step 200 (`21.53 → 21.82 → 22.05 → 22.12`) puis redescente step 250 (`21.96`). Pattern "warmup-overshoot" classique bf16 sans loss scaler, pas une divergence catastrophique. Loss converge en ~250 steps après le hump.
- **vs fp32 (job 5075686, même setup)** : **+25% throughput** (80.0 vs 64.0 img/s). ETA 10M -20% (34.7h vs 43.4h). Mais fp32 avait loss monotone descendante dès step 50 → bf16 fait sa convergence un peu plus tard.
- **Verdict** : 🟢 throughput gain confirmé, 🟡 convergence à valider sur run plus long (>1000 steps) pour exclure dérive long-term. Acceptable pour bench. Pour Phase 1 prod : envisager hybride bf16 forward + fp32 master copy AdamW si convergence reste problème.
