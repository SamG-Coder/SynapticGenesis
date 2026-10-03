"""Run the existing native trace diagnostic, then summarize its emitted arrays.

Native C++/CUDA computes the model. Python only reads the checkpoint weights and
summarizes diagnostic files; no learning, generation, or checkpoint write occurs.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from activity_metrics import activity
from native_experiment import NativeCommands, sha, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from probes_cli import Reference


def inspect(checkpoint, source, out, diagnostic):
    checkpoint, source, diagnostic = [p.resolve() for p in (checkpoint, source, diagnostic)]
    raw = source.read_bytes()
    if not 2 <= len(raw) <= 4097 or b'\x1e' in raw:
        raise ValueError('Supply one document window of 2 to 4097 bytes without a document separator')
    if out.exists():
        raise ValueError('Activity inspection requires a fresh output directory')
    protocol = dict(checkpoint_sha256=sha(checkpoint), source_sha256=sha(source),
                    diagnostic_sha256=sha(diagnostic), checkpoint=checkpoint.as_posix(),
                    source=source.as_posix(), diagnostic=diagnostic.as_posix(),
                    input_bytes=len(raw), observed_steps=len(raw)-1,
                    math='Existing native strict FP32 trace diagnostic, reset recurrent state',
                    emission_epsilon=1e-6,
                    script_sha256={p: sha(p) for p in ('scripts/activity_metrics.py',
                        'scripts/inspect_native_activity.py', 'scripts/native_experiment.py',
                        'tests/probes_cli.py', 'tests/learned_trace_dump.cu')},
                    native_source_sha256={p.as_posix(): sha(p) for p in sorted(Path('src').glob('*.cu'))
                                           + sorted(Path('src').glob('*.cuh'))})
    out.mkdir(parents=True)
    write(out / 'protocol.json', protocol)
    native = NativeCommands(diagnostic, out)
    native(checkpoint, source, out / 'trace')
    reference = Reference(checkpoint)  # Parameter decoding only; no CPU forward.
    if reference.cell not in (5, 6):
        raise ValueError('Native trace diagnostic supports selective and associative cells')
    steps = len(raw) - 1

    def array(name, shape):
        value = np.fromfile(out / 'trace' / name, dtype='<f4')
        if value.size != int(np.prod(shape)) or not np.isfinite(value).all():
            raise ValueError('Malformed native diagnostic array: ' + name)
        return value.reshape(shape)

    logits, losses = array('logits.f32', (steps, 256)), array('losses.f32', (steps,))
    # Independently check that diagnostic losses correspond to these logits and
    # this exact byte sequence. This does not replace native-vs-CPU model checks.
    scores = logits.astype(np.float64)
    maxima = scores.max(axis=1)
    expected = maxima + np.log(np.exp(scores - maxima[:, None]).sum(axis=1))
    expected -= scores[np.arange(steps), np.frombuffer(raw[1:], dtype=np.uint8)]
    error = float(np.abs(expected - losses).max())
    if error >= 3e-5:
        raise ValueError('Native loss/input/logit consistency failure: ' + str(error))
    layers = []
    for index, block in enumerate(reference.blocks):
        prefix = f'layer-{index}-'
        shape = (steps, reference.h)
        arrays = {name: array(prefix + name + '.f32', shape)
                  for name in ('u', 'spikes', 'emission', 'z', 'gate')}
        array(prefix + 'norm.f32', (steps, reference.c))
        layers.append(dict(layer_zero_based=index,
                           **activity(arrays['spikes'], arrays['emission'], arrays['u'],
                                      block[3].numpy(), protocol['emission_epsilon'])))
    identities = {path.name: sha(path) for path in sorted((out / 'trace').iterdir())}
    for path, expected_hash in ((checkpoint, protocol['checkpoint_sha256']),
                                (source, protocol['source_sha256']),
                                (diagnostic, protocol['diagnostic_sha256'])):
        if sha(path) != expected_hash:
            raise ValueError('Input changed during inspection: ' + str(path))
    if any(sha(p) != expected_hash for p, expected_hash in protocol['script_sha256'].items()):
        raise ValueError('Diagnostic source changed during inspection')
    if any(sha(p) != expected_hash for p, expected_hash in protocol['native_source_sha256'].items()):
        raise ValueError('Native source changed during inspection')
    result = dict(protocol=protocol, cell_version=reference.cell, channels=reference.c,
                  hidden=reference.h, layers=layers, native_commands=len(native.commands),
                  mean_loss_nats_per_byte=float(losses.astype(np.float64).mean()),
                  independent_loss_consistency_max_error=error,
                  input_checkpoint_and_diagnostic_unchanged=True, trace_sha256=identities,
                  limits=[
                      'A finite reset-state observation window, not a live-stream firing distribution.',
                      'Native float32 execution only; no claim of CPU model parity from these statistics.',
                      'The direct-output proxy excludes cancellation, residual and associative paths, and task effects.',
                      'Participation ratio describes centered emission diversity, not task capacity or plasticity.',
                      'Zero-energy fractions and zero-variance participation ratios are undefined and reported as null.',
                      'No pruning, renewal, weight update or biological-age decision is made.'])
    write(out / 'result.json', result)
    print(out / 'result.json', 'native activity inspected; model unchanged')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path,
                        default=Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe'))
    args = parser.parse_args()
    inspect(args.checkpoint, args.input, args.out, args.diagnostic)
