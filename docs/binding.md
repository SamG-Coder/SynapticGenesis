# Binding facts to the queried object

The first location probes always asked about the first named object. [Binding-v2](../data/lessons-binding-v2.json) removes that shortcut: each group contains two location assignments and two questions, one about each object. The model must answer all four correctly. Sentence order, subject-first/location-first wording and question/request wording are balanced.

The selected templates create 96 single-object prerequisite lessons and 1,728 two-object training lessons. Development and test each contain 576 items in 144 groups. Unordered object pairs are held out, so both orders and all wording/location/query variants stay in one partition. All individual words and syntax are familiar; the task tests new object combinations within this controlled language, not open-world reasoning.

```powershell
python scripts/prepare_binding.py --reading data/prepared/development-v2-final/through-stage-4.dat
.\build\synapticgenesis.exe live --curriculum data/prepared/binding-v2/curriculum.sg --out runs/binding-lif --cell lif --seed 1337 --lr 0.0003 --chunk 128 --replay reservoir --replay-capacity 1024 --replay-every 4 --graph --speak-every 500 --tokens 96 --prompt "The bird " --fast --validation data/prepared/development-v2-final/13853.txt --eval-batches 32
.\build\synapticgenesis.exe language-probes --checkpoint runs/binding-lif/latest.ckpt --probes data/prepared/binding-v2/development.sgprobe --output runs/binding-lif/development.json
```

This schedule starts from random weights: 6,000 observations on approved reading, 4,000 on single-object prerequisites, then 24,000 on two-object binding. The later stages use the measured quarter base rate and answer emphasis of 64, preserving replay. A supplied `--checkpoint` is an explicitly different continuation experiment.

## Group protocol and verification

`SGPROBE2` uses the same item records as `SGPROBE1`, but each group ID must occur four times. The native parser requires a complete two-context/two-query grid, identical candidate strings and skill, opposite labels for the two queries in each context, and reversed labels for each query across contexts. Duplicate cells, inconsistent choices, missing items and invalid label patterns are rejected. The legacy item field named `pair` identifies the group; the report identifies `group_size: 4`, `groups` and `joint_accuracy`. Old two-item reports retain their original paired fields and also expose the generic joint metrics.

The same answer-only strict-FP32 scoring, greedy generation and context-erasure control apply. Every answer begins with independent recurrence and model parameters/moments stay unchanged. Joint accuracy requires all four candidate decisions; greedy exact joint accuracy requires all four unrestricted strings. Ignoring the query and copying either the first or last location achieves 50% item accuracy and zero joint accuracy in this balanced set.

`python tests/binding_cli.py --out runs/binding-cli-test` independently parses the facts and query to verify every source-derived label, checks the copying controls, validates partition membership and reproducible bytes, compares native LIF/ALIF scores and greedy output with the CPU oracle, and rejects malformed groups. The test validates the reserved test labels as data, but does not evaluate a trained model on them.

## Initial result

The earlier answer-emphasized founder scored 49.48% item accuracy and 0.69% joint accuracy on this harder development task. Training the default LIF model from scratch for the complete 34,000-observation schedule produced 50.17% item accuracy and zero joint accuracy, with 50% greedy exact answers. Its earlier-reader loss was 2.65942. This failed learning result motivates testing the temporal representation; repeating familiar answer formats has not established object binding. The reserved test remains unused.
