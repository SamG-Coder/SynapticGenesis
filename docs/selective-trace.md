# Input-dependent trace retention

The experimental `--cell selective` changes how the spike trace is stored. Its gate controls how much previous trace survives each byte and how much the new spike replaces it. This tests a different mechanism from `--cell gated`, whose gate changes the emitted readout. The [equations and backward rule](architecture.md#input-dependent-trace-retention) describe the complete implementation.

The two cells have the same 1,716,736 parameters at C256/H512/L4, the same state/cache sizes and identical parameter initialization for a given seed. Zero gates recover the ordinary trace cell. Retention stays in [0,1], giving a convex mixture of the earlier trace and current signed spike. A diagnostic verifies that a nonzero retention gate changes the first layer's stored trace while the read gate leaves that layer's trace unchanged.

The distinction is motivated by the temporal selection discussion in [Mamba](https://arxiv.org/html/2312.00752v2#S3.SS5) and earlier [LSTM forget-gate research](https://sferics.idsia.ch/pub/juergen/FgGates-NC.pdf). This is an original, bounded engineering experiment over our spiking model, not a reproduction of either architecture or a claim of biological equivalence.

## Validation and runtime

Architecture version 5 has its own checked identity. Equal-sized read-gate and retention-gate checkpoints cannot be interchanged. The same live engine supports graph generation, replay, optional SI, saved curriculum history and later lesson admission. Population inheritance copies gate parameters, bounded width growth is initially function-preserving, resource accounting includes all buffers, and old-age death still blocks population learning.

The CPU autograd oracle uses nonzero input-gate weights and signed incoming traces to exercise temporal gradient propagation and the gate's input-gradient path. A second fixture checks answer-emphasized targets. Native tests also compare streamed/chunked/graph inference, zero-gate trace equivalence, disabled-trace LIF equivalence, state/optimizer/speech restarts and replay isolation. The continuous output rejects ternary spike-add and the experimental indexed-spike paging path.

The neuron remains an optional experiment. The default LIF model and existing checkpoint meanings are preserved. Passing numerical and lifecycle checks is not evidence of language competence.

## Matched language and exposure protocol

The selected binding-v2 material is unchanged. Each model starts from scratch with the same seed, reading for 6,000 online observations and receiving 4,000 single-object prerequisite observations. The final two-object stage is declared at birth to last 120,000 observations. Measurements at 34,000, 67,000 and 130,000 total observations compare 24,000, 57,000 and 120,000 binding observations. This tests longer exposure alongside the cell change, because the previous rate and replay experiments did not establish that the earlier exposure budget was sufficient.

The base learning rate is 0.0003, the lesson-stage multiplier is 0.25, answer emphasis is 64, and a 1,024-window reservoir supplies one replay every four observations. Graph speech emits 96 bytes every 500 observations. Both cells receive identical source targets, replay descriptors and generated-byte counts at each measured point. Their generated content can differ and enters their own continuing recurrence, as in the established live protocol. Checkpoints are saved every 5,000 observations and at evaluation endpoints; this save cadence is shared by both arms.

Training-group fit and development-group generalization use the complete four-question binding assay. Earlier-reader loss uses the same independent reader and evaluation batches. The reserved test partition is never scored. Each comparison is one seed; the script allows declared independent repetitions. Time includes the live loop and checkpoint I/O, excluding process setup, probe scoring and initial/final validation.

```powershell
python scripts/selective_experiment.py --out runs/selective-binding-panel
```

The script preserves all three measured checkpoints, source identities, initial-array equality checks, sessions, response-sensitivity audits and seven-round strict-FP32 decode checks. Experiment results must be reported together with any failures; the architecture is not promoted solely on the basis of one favorable score.
