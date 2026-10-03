# Serial GPU acceptance for the membrane candidate

The [policy integration](membrane-policy.md) has compile and host evidence.
This stage adds a bounded native acceptance run, with its command sequence
declared before GPU execution. It makes no language or performance claim and
does not promote checkpoints into teaching or reproduction.

The runner holds a Windows process handle for the **whole 411M Python study
driver**. That driver continues after its native learning child exits, while
it performs the saved-checkpoint assessments. A process name, lock file,
elapsed timeout, or missing observation is not used as completion evidence.
The saved executable path and creation time must match when the runner opens
the handle; a reused PID does not satisfy the gate. Nonzero exit stops the run.

After that handle exits successfully, a read-only verifier checks the saved
study result against its declared protocol. It verifies all eight assessment
endpoints, all 80 assessment command records, original baseline equality,
checkpoint hashes and state, exposure matching, comparison deltas, the founder's
training command and full 216,289-update completion. It reads large checkpoints
with streaming hashes. This completion gate does not require the 411M model to
beat the 105M baseline; the independent mechanism experiment remains useful if
larger capacity fails to improve language.

The acceptance plan pins its scripts, native source, executables, generated
CTest configuration and predecessor protocol. The launched command also carries
the plan's SHA-256. Inputs are checked again after waiting and before each
command. A changed input stops execution instead of silently rebuilding or
using another checkpoint.

Twenty top-level commands run sequentially:

1. The candidate native probe generates eight membrane-gradient fixtures and
   runs eight exact save/resume cases with replay, curriculum changes, graph
   speech and optional SI.
2. Eight independent CPU autograd checks consume those native fixtures, covering
   four traced cell types and both single-sequence and batched gradients.
3. All 18 registered native CTest cases run with one worker. The JUnit result
   must contain every expected case without failures or skipped cases.
4. Six CPU autograd checks consume the newly generated ordinary cell fixtures.
5. The existing ragged-stream compatibility suite compares the preserved and
   candidate executables across six cells and FP32/TF32: 48 native commands
   check full checkpoint equality, speech equality and candidate restart.
6. The existing teacher CLI suite runs on each executable. Both suites must
   pass, and the sets and bytes of their checkpoint and transcript artifacts
   must match. This covers disabled-policy teacher/curriculum behavior.
7. An enabled-policy associative learner trains with two frozen teachers,
   stage replay, answer weighting, SI and graph speech. A separate-process
   restart must reproduce the complete checkpoint and transcript exactly.
   A matched zero-cost control must produce different learned weights, and
   evaluation/generation must preserve the checkpoint. These are mechanism
   checks, not evidence of better language or useful inherited knowledge.

All new model inputs in these mechanism tests are disposable synthetic
fixtures. They do not become admitted language sources. Timing differences in
these tests are not treated as speed measurements.

The runner records the command before launching it, captures stdout/stderr,
and stops on the first failure. Partial files and the failure record remain
available for diagnosis. A failed or interrupted directory is never restarted
automatically. Every gradient oracle and compatibility result must pass before
the final acceptance result is written.

The [host checks](../reports/membrane-acceptance-host.json) exercise malformed
plans and results, forbidden early dispatch, a mocked successful dispatch,
and real short CPU-only Python processes with zero and nonzero exit codes.
Mocked dispatch is explicitly not device evidence. A separate
[baseline verification report](../reports/membrane-acceptance-baseline.json)
records the 40 already available baseline assessment commands checked before
the 411M run finishes; it is not a completed-study result.

Example declaration after building and verifying the candidate binaries:

```powershell
python scripts/membrane_acceptance.py declare --out runs/membrane-acceptance-v1 --workspace D:/SynapticGenesis --predecessor D:/SynapticGenesis/runs/prose-411m-v1 --old D:/SynapticGenesis/build/validated-shared-views/synapticgenesis.exe --wait-pid DRIVER_PID --wait-executable DRIVER_EXECUTABLE
python scripts/membrane_acceptance.py run --out runs/membrane-acceptance-v1 --protocol-sha256 DECLARED_PLAN_SHA256
```

Use an observed live driver identity, never a guessed PID. If the predecessor
has already completed, omitting the wait arguments requires its saved result
to pass verification during declaration as well as execution. Actual dispatch
and device results are tracked separately from this host-verified runner.
