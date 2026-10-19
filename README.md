# Goblet

Goblet couples the GARGOYLES graph-based molecular generator with Boltz-2
affinity predictions. The repository includes the generator checkpoints and
the protein/MSA inputs used for the eight-target experiment.

## Reproducibility snapshot

The representative calculation is defined as follows:

| Item | Value |
| --- | --- |
| Targets | `akt1`, `ampc`, `cxcr4`, `cp3a4`, `hivrt`, `hivpr`, `gcr`, `kif11` |
| Independent runs | 5 per target (SGE task IDs 1--5) |
| Initial molecule | benzene (`c1ccccc1`) |
| Reward | `ASBoltz` (Boltz-2 affinity with SA-score filtering and a target-specific pocket constraint) |
| Coarse/fine MCTS steps | 1000 / 20 |
| Generator checkpoint | epoch 20 (`ckpt/pretrain/ver2/*-ep20.pth`) |
| Boltz package | 2.2.1 |
| Boltz inference seed | 42 (set in `Utils/boltz_api_constraints.py`) |

The generator also samples stochastically. GPU kernels and sampling therefore
mean that a repeated run is not expected to reproduce the CSV bit-for-bit; it
reproduces the reported inputs, model versions, reward, and search budget.

## Requirements

- Linux and an NVIDIA GPU supported by CUDA 12.1
- Conda
- Sun Grid Engine (SGE) for the supplied job scripts
- Boltz-2 model files `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, and the Boltz
  molecule cache. By default these are read from `${HOME}/.boltz` and copied
  to node-local storage for each job.

Create the pinned environment:

```bash
conda env create -n goblet -f env.yaml
conda activate goblet
```

`env.yaml` pins Python 3.10.14, PyTorch 2.2.1/CUDA 12.1, DGL 2.1.0,
RDKit 2025.9.3, and Boltz 2.2.1. The job script additionally records the
installed runtime versions and SHA-256 hashes of the generator and Boltz
checkpoints in a per-run metadata file.

Verify the versioned models and inputs before submitting work:

```bash
bash reproducibility/verify.sh
```

See [`reproducibility/README.md`](reproducibility/README.md) for the input map,
configuration details, output schema, and a description of the execution flow.

## Run one representative target

The script creates the required workspace from the versioned YAML and MSA, so
no manual preparation of `wkdir/` is needed. This submits five runs for KIF11:

```bash
qsub -N kif11 -v TARGET=kif11,GOBLET_ENV=goblet 0_generate.job.sh
```

The T4 cluster modules used in the study are loaded when the `module` command
is available. On another SGE cluster, load an equivalent CUDA 12.1 toolchain
before submission or edit the module stanza.

Useful submission overrides are `INIT_SMILES`, `N_STEP_COARSE`,
`N_STEP_FINE`, `GOBLET_ENV`, `BOLTZ_ENV`, `SOURCE_BOLTZ_CACHE`, and
`BOLTZ_CACHE_DIR`. `BOLTZ_ENV` defaults to `GOBLET_ENV`; set it separately if
the generator and Boltz server are installed in different Conda environments.
For example, a short pipeline check can be submitted with:

```bash
qsub -N kif11 \
  -v TARGET=kif11,GOBLET_ENV=goblet,N_STEP_COARSE=20,N_STEP_FINE=20 \
  0_generate.job.sh
```

Results are written to `data/result/<target>_<task>.ASBOLTZ.csv`. The matching
`.meta.txt` and `.api.log` files record provenance and the Boltz server log.
Existing result files are never overwritten.

## Run all eight targets

`batch.sh` submits the complete 8 x 5 job matrix. `GROUP` and `MAIL` are
optional SGE submission settings:

```bash
GROUP=my-sge-group MAIL=name@example.org GOBLET_ENV=goblet bash batch.sh
```

To omit either option, leave it unset.
