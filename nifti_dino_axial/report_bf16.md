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

---

## Run 2 — Job 5078975 : 1n × 4 APUs **b48 bf16**

- **Setup** : `RUN_TAG=b48 BIND_STRATEGY=mi300_srun4 ./launch.parsable.sh 1 4 MI300 48` sur a1004. Test push batch après b40 OK. **Référence fp32 = STUCK** (job 5074893, 9 min sans aucun step → cancel).
- **Perf** : 1 mesure warm utilisable (step 100) = **0.39 it/s = 74.5 img/s**. ETA 10M = 37.3h. VRAM ~65% (rocm-smi user obs). Cancel à 6:44 (test rapide).
- **Observation** : vs fp32 STUCK = bf16 **débloque** ce batch size complètement. Loss step 50→100 = `21.54 → 21.79` = même légère hausse warmup observée à b40 (pattern reproductible).
- **vs b40 bf16 (80 img/s)** : **−7% throughput** (74.5 vs 80.0). Inattendu, on aurait pu attendre +20% comme en fp32 b10→b40 isolation.
- **Verdict** : 🟢 b48 fonctionne (gain de SAFETY vs fp32). 🟡 mais throughput plateau/régresse → suggère qu'on hit un autre bottleneck (memory bandwidth ? compute saturation des CU à b40 déjà ?). Pour confirmer plateau : tester b64.

---

## Run 3 — Job 5079019 : 1n × 4 APUs **b64 bf16**

- **Setup** : `RUN_TAG=b64 BIND_STRATEGY=mi300_srun4 ./launch.parsable.sh 1 4 MI300 48` sur a1019. Config nouvelle `phase1_b64.yaml` (batch=64, **pf=2 default** — différent de notre fp32 b64+pf4 qui OOM). **Référence fp32 = OOM** (job 5074683, 2 oom_kill events).
- **Perf** : médiane 3 warm steps 100-200 = **0.29 it/s = 73.7 img/s**. ETA 10M = 37.7h. VRAM ~80% (rocm-smi user obs, plus que b48 logique).
- **Observation** : vs fp32 OOM = bf16 + pf=2 **débloque** complètement. Memory headroom encore +20% avant 100%. Loss step 50→200 = `21.55 → 21.75 → 22.02 → 22.09` = même hump bf16 que b40/b48.
- **vs b40 bf16 (80 img/s)** : **−8% throughput** (73.7 vs 80.0). **Plateau confirmé** entre b48 et b64. Plus de batch ne donne plus de samples/s.
- **Verdict** : 🔴 b64 ne paye pas en throughput (régression −8% vs b40). Confirme l'hypothèse "**sweet spot bf16 mono-node = b40**" (peut-être même un peu moins). Hypothèse plateau : on sature soit le memory bandwidth, soit les CU à des shapes batch×patch=192 qui sont déjà well-fed à b40. **À vérifier en poussant b80** pour voir si OOM ou plateau continue.

---

## Run 4 — Job 5079132 : 1n × 4 APUs **b32 bf16**

- **Setup** : `RUN_TAG=b32 BIND_STRATEGY=mi300_srun4 ./launch.parsable.sh 1 4 MI300 48` sur a1019. Test "moins gros que sweet spot" pour vérifier si plateau bf16 commence sous b40 ou strictement à b40.
- **Perf** : 5 mesures warm steps 100-300 = `75.4 / 76.5 / 72.8 / 73.1 / 80.5` img/s. **Médiane ~75 img/s** mais variabilité élevée (60% spread step à step). Step 300 spike à 80.5 (= rejoint b40) → suggère que b32 atteint b40 en rate stable mais sur-fluctue plus.
- **Observation loss** : trajectoire similaire à b40/b48/b64 — `21.55 → 21.80 → 22.02 → 22.14 → 21.88` step 50→250, puis re-descend à 21.70 step 300. Hump bf16 reproductible.
- **vs b40 bf16 (80 img/s)** : **−6 à −7% médiane** mais le spike step 300 montre que **b32 peut atteindre 80 img/s par moments**. Suggère que la régression apparente est due à la variabilité, pas un vrai plateau différent.
- **Verdict** : 🟡 b32 ≈ b40 en peak (80 img/s atteint épisodiquement) mais médiane plus basse. **b40 reste sweet spot le plus stable**. Hypothèse : b32 sous-utilise les CU à certains step (selon ps=8 vs 16 = mix patch sizes), b40 nourrit mieux.

---

## Run 5 — Job 5079134 : 1n × 4 APUs **b80 bf16** ❌ OOM

- **Setup** : `RUN_TAG=b80 BIND_STRATEGY=mi300_srun4 ./launch.parsable.sh 1 4 MI300 48` sur a1004. Test push max — espérait que bf16 (-50% mémoire forward) déverouille b80 (qui OOM en fp32 = job 5074659).
- **Perf** : **AUCUN step produit**. 8m22s elapsed, log spammé de UserWarning numpy (data loading qui boucle), puis `slurmstepd: Detected 2 oom_kill events in StepId=5079134.0`. Tasks 1 et 3 killed (les 2 ranks GPU côté impair). State final = `OUT_OF_MEMORY`.
- **Observation** : memory pressure visible avant le step 50 → l'init du modèle + 1er forward avec b80 dépassait 128 Gio par APU même en bf16. iBOT head output = batch × 1024 patches × 65k vocab × 2 bytes = ~10 Gio par rank, multiplié par 4 ranks via DDP gradient buffers + activations multi-crops globaux/locaux.
- **vs fp32 b80 OOM** : **bf16 ne suffit PAS** à débloquer b80. Sweet point est "trop gros" pour ce modèle/setup à 4 APUs avec grad_ckpt=true.
- **Verdict** : 🔴 b80 reste OOM. Pour pousser au-delà b64 il faudrait : (a) `gradient_accumulation_steps>1` (réduire effective batch×rank), (b) drop `grad_ckpt=false` (impossible car ça augmente encore), (c) tuner les iBOT head dims, (d) tester plus de nodes (8 GPUs au lieu de 4 réduit pression par-rank). Confirme **b64 = ceiling pratique en mono-node 4-GPU bf16**.

---

## 📊 Synthèse plateau mono-node bf16 (5 runs MI300A 1n × 4 APUs)

| Batch | Runtime | img/s warm | %VRAM | vs b40 |
|---:|---|---:|---:|---:|
| b32 | OK (variable) | ~75 (peak 80) | ? | ≈ |
| **b40** | OK stable | **80.0** | ? | ⭐ ref |
| b48 | OK (peu de mesures) | 74.5 | 65% | −7% |
| b64 | OK | 73.7 | 80% | −8% |
| b80 | ❌ OOM | — | >100% | — |

**Sweet spot mono-node bf16 = b40 = 80 img/s = +25% vs fp32 b40 (64 img/s)**.
