# SciSilentBench Final Results Summary

## Scope and validation

The original `tasks_20.json` was not limited to UNIT bugs. It contained four validated categories: TIME_SCALE (1), RAD_DEG (11), JD_MJD (4), and UNIT (4); FRAME was missing. The updated mutator now creates `tasks_25.json` with 25 tasks covering all five promised categories: TIME_SCALE (1), RAD_DEG (11), JD_MJD (4), UNIT (4), and FRAME (5). The mutator validates that clean code passes both checks and that each mutated case passes its visible test while failing its hidden oracle.

The set remains imbalanced and template-heavy: 11/25 cases are angle-conversion variants, while TIME_SCALE has only one case. It supports descriptive category ablations, not broad claims about scientific software in general.

## Models and conditions

Both models were run locally through Ollama using quantized GGUF checkpoints:

- E2B: `hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M`
- E4B: `hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M`

Both conditions ran on all 25 task IDs per model, with a 180-second task timeout. E2B uses the earlier completed 20-task records plus a separate five-task FRAME evaluation, with the same model, agent code, prompts, and grader. E4B ran the full 25-task set under each condition.

## Overall results

| Model | Condition | TFR | SFR | Abstention | AUROC | ECE | Mean tokens | Mean time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q4_K_M | A | 68.0% | 0.0% | 28.0% | 0.934 | 0.044 | 1166.0 | 109.73 |
| Gemma 4 E2B Q4_K_M | B | 4.0% | 0.0% | 96.0% | 1.000 | 0.000 | 1033.0 | 124.90 |
| Gemma 4 E4B Q4_K_M | A | 20.0% | 0.0% | 80.0% | 1.000 | 0.000 | 943.3 | 151.71 |
| Gemma 4 E4B Q4_K_M | B | 0.0% | 0.0% | 100.0% | 0.500 | 0.000 | 50.0 | 176.64 |

## Table 3: TFR and SFR by bug type

| Bug type | n | E2B A TFR/SFR | E2B B TFR/SFR | E4B A TFR/SFR | E4B B TFR/SFR |
|---|---:|---:|---:|---:|---:|
| FRAME | 5 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| JD_MJD | 4 | 100.0% / 0.0% | 0.0% / 0.0% | 50.0% / 0.0% | 0.0% / 0.0% |
| RAD_DEG | 11 | 81.82% / 0.0% | 9.09% / 0.0% | 27.27% / 0.0% | 0.0% / 0.0% |
| TIME_SCALE | 1 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| UNIT | 4 | 100.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

## Failure analysis (observed source and outcome)

1. **TIME_SCALE, LK-042 — E2B abstained.** The code uses UTC Julian dates against a TDB reference epoch:

   ```python
   dt = time_obj.utc.jd - t0_tdb_jd
   return dt % period_days / period_days
   ```

   It should compute the phase using `time_obj.tdb.jd` to match `t0_tdb_jd`. Condition A abstained; Condition B timed out and abstained.

2. **RAD_DEG, ANG-002 — E2B Condition B abstained.** The function returns the degree input unchanged despite its radians contract:

   ```python
   def detector_roll_rad(roll_deg: float) -> float:
       return float(roll_deg)
   ```

   The repair is `float(np.deg2rad(roll_deg))`. The positive-only visible test permits this physically incorrect value; the hidden numeric oracle catches it.

3. **JD_MJD, JDM-001 — E2B Condition B abstained; E4B Condition A timed out.** The mutated function returns Modified Julian Date while the prompt requires Julian Date:

   ```python
   def observation_julian_date(time_obj: Time) -> float:
       return float(time_obj.mjd)
   ```

   The correct result is `float(time_obj.jd)`. E2B Condition A repaired all four JD_MJD cases, but the other model/condition results show that this performance does not generalize across prompts and conditions.

4. **FRAME, FRM-004 — E2B Condition A submitted an invalid repair.** The buggy implementation returns the coordinate in its native GCRS frame:

   ```python
   return float(coord.gcrs.ra.deg)
   ```

   The submitted patch constructed `ICRS(obstime=obstime)` and returned native `.ra`, instead of transforming the original GCRS coordinate with `coord.icrs.ra.deg`. Its actual body was:

   ```python
   def icrs_ra_from_gcrs_4(ra_deg, dec_deg, obstime):
       coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                        frame=ICRS(obstime=obstime))
       return float(coord.ra.deg)
   ```

   The submission failed the visible test, so it was not counted as a silent failure. All five FRAME cases had 0% TFR for both models and conditions; the E2B conditions mostly abstained or timed out.

## Interpretation

Condition B produced no silent failures in this run, but its abstention rate was 96% for E2B and 100% for E4B. That safety result came with very few submitted repairs. Larger E4B did not improve TFR: under Condition A it achieved 20%, compared with 68% for E2B, and under Condition B it achieved 0%, compared with 4% for E2B. E4B was also slower on average. These two quantized model points do not establish a model-size scaling law; they show no positive scaling trend in this setup.

AUROC/ECE should be interpreted cautiously. E4B Condition B submitted no fixes, so AUROC defaults to 0.5 and ECE=0.0 does not indicate useful calibration. E2B Condition B's AUROC=1.0 is based on only one true fix. The set is small, synthetic, and strongly imbalanced by category.

## Errors and reproducibility notes

- The official Ollama E4B model pull returned HTTP 401. The Ollama-compatible Hugging Face Q4_K_M checkpoint downloaded successfully, appeared in `ollama list`, and answered a smoke prompt.
- The machine has 16 GB RAM and no detected NVIDIA GPU. The promised 26B MoE and 31B models were not run; the scaling claim in any manuscript must be limited to the tested E2B and E4B checkpoints.
- E4B Condition B emitted LiteLLM connection-timeout traces and an async coroutine warning. The evaluator nevertheless completed and saved 25 records; those model timeouts are included as abstentions in the reported metrics.
- ADK emitted its experimental JSON-schema tool warning. The runs completed.
- No SciSilentBench manuscript draft was present in the Gemma project folder. `PAPER_FINAL.md` is a paper-ready replacement written from the available code and results, not an edit of an unavailable draft file.

## Result files

- `tasks_25.json`
- `results_E2B_A.json`
- `results_E2B_B.json`
- `results_E4B_A.json`
- `results_E4B_B.json`

## Independent Week 1 audit and protocol diagnostics

An independent rerun of the mutator checks verified every task: all 25 clean implementations passed both visible and hidden checks, and all 25 mutated implementations passed the visible check while failing the hidden oracle (100/100 expected outcomes). Counts were TIME_SCALE 1, RAD_DEG 11, JD_MJD 4, UNIT 4, FRAME 5. The 25 task IDs in each result file matched the task set, and independently recomputed overall metrics matched each saved summary.

A trajectory audit separated outer timeouts from missing tool decisions. This does not alter the published aggregate abstention metric, which combines them, but shows that the condition B values are not evidence of deliberate abstention:

| Run | Outer timeouts | Ended without decision-tool call | Used tool but no final decision | Submitted | Explicit abstain calls |
|---|---:|---:|---:|---:|---:|
| E2B A | 5 | 2 | 0 | 18 | 0 |
| E2B B | 9 | 14 | 1 | 1 | 0 |
| E4B A | 12 | 5 | 3 | 5 | 0 |
| E4B B | 24 | 1 | 0 | 0 | 0 |

The evaluator currently collapses outer timeouts and missing tool decisions into ABSTAIN. Condition B’s observed abstention rate therefore primarily reflects inference timeouts and failure to use the submit/abstain interface, rather than explicit decisions to abstain. The E4B B run is particularly infrastructure-limited (24/25 timeouts), so it should not be interpreted as a clean test of invariant reasoning. A follow-up protocol comparison should report these failure classes separately and hold model, task IDs, timeout, and generation settings fixed.

The Week 1 oracle audit found no failing task checks. The five FRAME cases passed the same clean/mutated test pattern as the other categories; the earlier FRM-004 attempted repair remains a failed visible test, not a silent failure.

## Reproducibility/security maintenance

The independent audit command used the project virtual environment and completed all 100 task checks. `run_all.py` had a hard-coded API credential; the literal has been removed and the script now reads `GOOGLE_API_KEY` or `GEMINI_API_KEY` from the environment. The exposed credential should be revoked by its owner. The script compiles after the change.

## Weeks 3–4: related work, scope, and final recalculation

The final related-work discussion distinguishes SciSilentBench from ScienceAgentBench’s broader scientific-discovery tasks, SWE-bench’s issue-linked repository repairs, AgentBench’s interactive environments, and scientific property-based testing. The paper makes a narrow complementarity claim: this pilot isolates physically meaningful defects that pass a permissive visible test but fail a hidden oracle. It does not claim to be the first or broadest scientific coding-agent benchmark.

No additional tasks were added to the scored set. The existing 25 are heavily parameterized synthetic snippets (RAD_DEG alone is 11/25); unreviewed additions without matching model runs would inflate the dataset without improving the evidence. Future expansion should use distinct scientific modules, more than one mutation mechanism per category, independent domain review, and matched experiments.

The final metric recalculation reproduced all saved overall and by-bug-type summaries exactly. Each result file contains 25 unique task IDs matching `tasks_25.json`. Final values remain:

| Model | Condition | TFR | SFR | Abstention* | AUROC | ECE | Mean tokens | Mean time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 E2B Q4_K_M | A | 68.0% | 0.0% | 28.0% | 0.934 | 0.044 | 1166.0 | 109.73 |
| Gemma 4 E2B Q4_K_M | B | 4.0% | 0.0% | 96.0% | 1.000 | 0.000 | 1033.0 | 124.90 |
| Gemma 4 E4B Q4_K_M | A | 20.0% | 0.0% | 80.0% | 1.000 | 0.000 | 943.3 | 151.71 |
| Gemma 4 E4B Q4_K_M | B | 0.0% | 0.0% | 100.0% | 0.500 | 0.000 | 50.0 | 176.64 |

*Abstention includes outer timeouts and trajectories that did not call a decision tool. Neither Condition A nor Condition B recorded an explicit `abstain` tool call. Thus the B results do not show an SFR reduction over A (both were already 0%) or demonstrate calibrated abstention. The E4B-B outcome is dominated by timeouts.

The final manuscript is `PAPER_FINAL.md`. It narrows the conclusions to the audited 25-task pilot, reports that the protocol comparison is inconclusive, and lists task imbalance, synthetic construction, missing model sizes, timeouts, and decision-tool compliance as limitations. Reproducibility artifacts and runtime versions are recorded in the paper’s Appendix B.
