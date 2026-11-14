# v4: Kaggle student training

The undefended student completed 834 optimizer steps across three epochs on a T4 GPU in FP16, using 4,448 examples. Training took 7,056 seconds; training loss was 0.08436 and mean token accuracy was 0.9916. Validation was disabled; these are not held-out results.

- Notebook checkout: `1e5b3d851a460b33524b9ee1c10c6a0035e1257b`. The training manifest does not record a code revision.
- Base model: `Qwen/Qwen3.5-2B-Base`, revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
- Training dataset SHA-256: `e90810fc40d07f5cc2ded31e2924385c6f3f198d3dde6492218259a2885d476d`.
- Final adapter SHA-256: `351036ef32109f440863dac7624f905b892bfbaccac0f76053e2446d80c4e1fb`.
- The sanitized state archive excludes the original root `.env`. Its SHA-256 is `8f7e0c98826e778dc71c2012d61dca03daaae32ecb37fb9ea92192e40cd6f2db`. Original archive SHA-256: `ca55371a7fc8a4138c656989d929a2ea9e417e0f899de653207db3ff96f75ad4`.

The `undefended/` directory contains the downloaded ledger, dataset, configuration, manifests, adapter, and checkpoints; `training.log` contains training output. Model weights, checkpoint binaries, and logs are gitignored and will not be preserved by a normal commit. The adapter requires the pinned base model. The archive contains other run data and should be reviewed before sharing.
