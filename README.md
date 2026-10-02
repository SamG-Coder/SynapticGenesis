# SynapticGenesis

**Small spiking language models that learn continuously and develop through a curriculum. Native C++17 and CUDA.**

SynapticGenesis starts founder models from random weights, using an explicit selection of reading material and a fixed byte vocabulary. The longer-term design is a population of models at different developmental stages: two parents contribute inherited model traits to a child, and teachers help it learn from selected source material. Each generation must demonstrate useful learning and retention.

The native learner, population registry, fitness-gated reproduction and bounded neuron growth work as research prototypes. Automatic stage promotion and teacher feedback are planned. Current models are small language-learning experiments; coherent conversation and general reasoning have not been established. Browser integration comes later.

## What runs today

- Signed leaky integrate-and-fire neurons, with an optional adaptive-threshold cell.
- Training and generation in one live process using shared weights and persistent neuron state.
- Bounded replay of previously observed source windows, adjustable core plasticity and optional synaptic-importance consolidation.
- CUDA graph decoding, checksum-protected checkpoints and restoration of optimizer, recurrent, replay and learning-history state.
- A selected general-reading curriculum with three stages, separate validation/test books and reproducible source hashes. Native live stage transitions retain replay and consolidation history.
- Optional selected science material and a native controlled retention experiment, with explicit stage repetition and matched new-source observations.
- Native parent selection, whole-block inheritance, inherited learning settings and probabilistic hidden-neuron growth within size limits.
- GPU-memory-based population credits, stricter selection under scarcity, inherited lifespans and old-age death.
- Population-owned live sessions update the exact checkpoint used for selection, preserve lifespan/lineage, and reject deceased members.
- Eight native numerical/runtime test suites, independent CPU gradient checks and a scalar consolidation oracle.

The default model has **1,186,304 parameters**, four residual blocks, width 256 and 512 spiking neurons per block. Training uses dense CUDA/cuBLAS operations, surrogate gradients and AdamW. Spikes do not by themselves establish an energy or speed advantage. See the [architecture](docs/architecture.md) and [development design](docs/general-development.md).

## Build

Requires an NVIDIA CUDA GPU, the CUDA toolkit, CMake and a C++17 compiler. The Windows build script locates Visual Studio C++ tools and Ninja. Local validation uses an RTX 5080, CUDA 13.3 and MSVC 19.51.

```powershell
git clone https://github.com/SamG-Coder/SynapticGenesis.git
cd SynapticGenesis
.\build.ps1
```

The script defaults to CUDA architecture `120`; pass `-Architecture` for another supported GPU. It builds `build/synapticgenesis.exe` and runs CTest. A normal CMake build is also available; other platforms have not been tested:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Prepare the foundation material

Python 3.10+ is used for source preparation and optional test tools. Model training and inference execute in C++/CUDA.

```powershell
python scripts/prepare_corpus.py
```

[data/sources.json](data/sources.json) is the allowlist. The preparation script obtains the selected text editions directly from Project Gutenberg, caches their original notices and catalogue pages, removes ebook framing, and writes `data/prepared/foundations-v1/manifest.json` with URLs and SHA-256 hashes. Digital text avoids unnecessary PDF OCR.

| Use | Selected material |
| --- | --- |
| Stage 1: words and simple sentences | McGuffey's Eclectic Primer and First Eclectic Reader |
| Stage 2: connected reading | McGuffey's Second Eclectic Reader |
| Stage 3: longer stories and explanations | McGuffey's Third Eclectic Reader |
| Validation | New National First Reader |
| Final test | The Beacon Second Reader |

Each `stage-N.dat` contains only that stage's training sources. `train.dat` pools all training stages. The `through-stage-N.dat` files and `curriculum.sg` provide cumulative editions for live development without discarding earlier replay windows. `validation.dat` and `test.dat` contain separate books and never supply training targets. Documents are separated by byte `0x1e`, which the model excludes from sampled windows. Exact normalized paragraphs of at least 120 characters are deduplicated, reserving held-out material first.

This is an initial, small historical reading curriculum. Its stages describe increasing text complexity, not validated human ages. Shared tales, paraphrases, shorter overlap and historical assumptions remain possible. Broader modern subject coverage and independent skill probes are future work. See [data details](docs/data.md).

## Optional batched founder training

Use fresh output directories. These commands start stage 1 from random weights and continue the same model through later material. `--steps` is the cumulative optimizer-update target; `--allow-new-corpus` explicitly permits the next stage's training data.

```powershell
.\build\synapticgenesis.exe train --data data/prepared/foundations-v1/stage-1.dat --validation data/prepared/foundations-v1/validation.dat --out runs/founder-stage-1 --steps 3000 --seed 1337 --fast

.\build\synapticgenesis.exe train --resume runs/founder-stage-1/latest.ckpt --allow-new-corpus --data data/prepared/foundations-v1/stage-2.dat --validation data/prepared/foundations-v1/validation.dat --out runs/founder-stage-2 --steps 6000

.\build\synapticgenesis.exe train --resume runs/founder-stage-2/latest.ckpt --allow-new-corpus --data data/prepared/foundations-v1/stage-3.dat --validation data/prepared/foundations-v1/validation.dat --out runs/founder-stage-3 --steps 9000

.\build\synapticgenesis.exe sample --checkpoint runs/founder-stage-3/best.ckpt --prompt "The bird " --tokens 240 --graph

.\build\synapticgenesis.exe evaluate --checkpoint runs/founder-stage-3/best.ckpt --data data/prepared/foundations-v1/test.dat --batch 16 --context 128 --batches 64
```

This is a manual batch schedule. Completing a stage's updates does not prove mastery. Each stage inherits weights and optimizer history; use the live curriculum below to retain an entire ongoing learning stream. Continuing batched training with a higher total step count extends its cosine learning-rate schedule. Compare ordered content with shuffled content at equal exposure before claiming a curriculum advantage.

## Learn and generate in one runtime

```powershell
# Begin with random weights and pause at the end of the first stage.
.\build\synapticgenesis.exe live --curriculum data/prepared/foundations-v1/curriculum.sg --out runs/live-founder --updates 2000 --lr 0.0003 --seed 1337 --replay reservoir --replay-capacity 1024 --replay-every 4 --graph --speak-every 500 --tokens 160 --prompt "The bird " --fast

# Resume the same individual through the remaining stages, retaining its history.
.\build\synapticgenesis.exe live --resume runs/live-founder/latest.ckpt --curriculum data/prepared/foundations-v1/curriculum.sg --out runs/live-founder --updates 6000 --prompt "The bird "
```

Observed text supplies the next-byte target. Generation reads the same mutable parameters and carries neuron state; generated text is not used as its own training target. Learning and speaking alternate at completed update boundaries. Replay uses separate recurrent state but the same weights and optimizer. See [runtime and checkpoint semantics](docs/architecture.md).

The schedule introduces new documents at fixed update counts while retaining earlier source windows for replay. It keeps weights, optimizer, speech RNG and optional consolidation history, records every transition, and archives each stage checkpoint. These are exposure stages; there is no automatic mastery decision. See the [live curriculum protocol](docs/live-curriculum.md), including how to prepare a fresh output directory if your earlier corpus lacks cumulative files.

Single-corpus `live --data` remains available. Resume requires the same source edition and prompt; curriculum resume also verifies the schedule and future source editions. Explicit `--lr`, `--replay-every` and `--si-strength` overrides change supported policy settings. For a curriculum, `--lr` is the base rate before the stage multiplier. A new `live --checkpoint` stream inherits weights and optimizer but resets stream/replay/consolidation history. The batch commands above also do not preserve the complete live history.

Optional `--consolidation si --si-strength 0.001` enables a synaptic-importance penalty. Its numerical behavior is tested; beneficial long-term retention has not been established on this curriculum. `--cell alif` creates a model with adaptive thresholds. Neither option is automatically better than the default. Neuron-indexed storage is an isolated benchmark, not the active training or generation storage path.

`metrics.jsonl`, `transcript.txt` and `session.json` record live runs. Placing a file named `STOP` in the run directory requests a checkpointed stop after a completed update; remove it before resuming. Runs are bounded by their requested update counts.

## Validate

`build.ps1` runs the eight native suites. For independent gradient checks, install CPU PyTorch and NumPy in your own test environment and run:

```powershell
python tests/oracle.py build/test-results
python tests/oracle.py build/live-test-results
python tests/oracle.py build/adaptive-test-results
python tests/synaptic_oracle.py build/synaptic-test-results
python tests/burn_policy.py --out runs/burn-policy-test
python tests/population_cli.py --out runs/population-cli-test
python tests/curriculum_cli.py --out runs/curriculum-cli-test
python tests/population_live_cli.py --out runs/population-live-cli-test
python tests/retention_cli.py --out runs/retention-cli-test
```

The consolidation oracle and CLI integration test use only Python's standard library. Numerical tests use disposable synthetic models, isolated from founders. [Validation evidence](reports/validation.md) records what was checked for this repository.

The [retention protocol](docs/retention.md) compares old-memory replay, extra current-chunk updates, consolidation and slower learning from the exact same starting model. Its [selected development sources](data/sources-development-v2.json) add a science training book and a different held-out geography book. They are an optional versioned extension; the original foundation source selection stays reproducible.

Across three seeds, replay plus a quarter later-stage learning rate gave the lowest old/new validation loss among seven tested settings. The explicit [reading-to-science profile](data/curricula/reading-to-science.sg) reproduces that development pattern through ordinary live learning; [results and commands](docs/retention.md#three-seed-result-and-usable-profile) include the controls, timing limits and remaining dialogue-quality gap. Newborns inherit parental base learning-rate traits independently of their parents' current stage slowdown.

## Development direction

The [development design](docs/general-development.md) specifies separate developmental stage and generation records, inheritance from two parents, teaching from selected source material, and selection against matched controls. The [native evolution commands](docs/evolution.md) register founders, evaluate eligibility, create children and inherit settings. Whole-block crossover can disrupt learned channel roles and every child must train and requalify; no improvement is assumed merely from birth.

Resource-sensitive breeding and lifespans are implemented. Available GPU memory sets the population budget; scarce resources narrow parent selection and raise the required improvement. Models die at their configured simulation lifespan, release population credits and remain archived on disk. `population-live` connects a member to the live curriculum, preserving its age and lineage while publishing learning checkpoints directly for the next selection round. Dead members cannot start another population learning session. Stage-specific mastery probes, teacher feedback, channel alignment, structural pruning and a town simulation remain future work.

## License and sources

Original project code is [MIT licensed](LICENSE). CUDA/cuBLAS and independently downloaded books retain their own terms. Raw source editions, prepared corpora, compiled binaries and checkpoints are excluded from Git. The repository publishes the selected source manifest, preparation tools, curriculum definitions and compact validation evidence. [Research references](docs/references.md) distinguish inspiration from implemented mechanisms.
