# Immediate resident conversation GPU validation

Executed on 2026-10-04 at the user's request, alongside the existing membrane study on the RTX 5080. This separate run does not replace or modify the queued 105M acceptance protocol.

All four 44,580-parameter native CUDA fixtures passed: ordinary FP32, ordinary TF32, membrane FP32 and membrane TF32. The run executed 28 reference commands, four resident script commands and one interactive stdin process. Full checkpoints matched through learning and restart; questions preserved the full training state; sampled answer bytes matched the reference. The interactive process remained alive between questions and learning, and recovered from a rejected command.

The machine-readable evidence is [resident-conversation-immediate-gpu.json](../reports/resident-conversation-immediate-gpu.json). Raw checkpoints and command logs remain in `D:/SynapticGenesis/runs/resident-conversation-immediate-v1`.

This validates the small-model GPU execution path. It does not establish useful language quality, 105M correction retention, or isolated performance. The concurrent training workload affected timing; do not use these timings as standalone throughput benchmarks. The full 105M comparison remains queued behind the original correction experiment.
