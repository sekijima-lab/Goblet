# Reproducing the eight-target Goblet calculation

## Versioned inputs

Each target uses a pair of files in `data/sample_yamls/`:

| Target | Protein input | MSA | Pocket contact |
| --- | --- | --- | --- |
| AKT1 | `boltz_akt1.yaml` | `msa_akt1.a3m` | A:87 |
| AmpC | `boltz_ampc.yaml` | `msa_ampc.a3m` | A:62 |
| CXCR4 | `boltz_cxcr4.yaml` | `msa_cxcr4.a3m` | A:131 |
| CYP3A4 (`cp3a4`) | `boltz_cp3a4.yaml` | `msa_cp3a4.a3m` | A:274 |
| HIV protease | `boltz_hivpr.yaml` | `msa_hivpr.a3m` | A:25 |
| HIV reverse transcriptase | `boltz_hivrt.yaml` | `msa_hivrt.a3m` | A:103 |
| Glucocorticoid receptor | `boltz_gcr.yaml` | `msa_gcr.a3m` | A:126 |
| KIF11 | `boltz_kif11.yaml` | `msa_kif11.a3m` | A:119 |

The contact definitions and benzene reference scores are declared in
`Utils/reward.py`. `0_generate.job.sh` copies the selected YAML to
`wkdir/<target>_<task>/input.yaml` and places its MSA beside it. The workspace
name is significant because the reward implementation uses it to select the
target-specific contact and reference score.

`reproducibility/checksums.sha256` covers all eight YAML/MSA pairs, the ZINC
fragment source, and the three epoch-20 generator checkpoints. Run
`bash reproducibility/verify.sh` from the repository root before submission.

Boltz-2 weights are distributed separately and are not committed here. Every
job writes their SHA-256 hashes to its metadata file.

## Execution flow

1. SGE assigns a target name and array task ID (1--5).
2. The job validates and stages the target YAML/MSA.
3. The Boltz cache is copied to node-local storage.
4. `Utils/boltz_api_constraints.py` loads the Boltz-2 confidence and affinity
   checkpoints once and exposes a loopback-only prediction endpoint.
5. `Generator/mcts.py` loads generator checkpoint epoch 20 and performs a
   1000-step coarse search with 20 fine steps per simulation.
6. `ASBoltz` rejects molecules with SA score above 3.5, applies the target
   contact constraint, and converts the Boltz affinity value to the reward
   relative to the target-specific benzene value.
7. The generated molecules are sorted by reward and saved as CSV.

The API uses one structure sample and three affinity diffusion samples. Its
confidence model uses three recycling steps; its affinity model uses five.
Both use 200 sampling steps. These settings are in
`Utils/boltz_api_constraints.py`.

## Output

The result CSV contains:

- `SMILES`: generated molecule;
- `Reward`: target-specific ASBoltz reward;
- `Sim`: Morgan-fingerprint Tanimoto similarity to the initial molecule;
- `Step`: MCTS step at which the molecule was generated;
- `Imp`: reward minus the initial-molecule reward.

The metadata file records the git commit, target, task, search settings,
package versions, GPU, input/checkpoint hashes, timestamps, and exit status.
The API log contains individual Boltz timings and errors.

## Expected resources and limitations

The reference job requests one GPU node for 24 hours. Five tasks are submitted
per target. Results can vary because generator sampling and some GPU
operations are stochastic. Boltz predictions are computational predictions;
an improved reward does not establish experimental binding or activity.
