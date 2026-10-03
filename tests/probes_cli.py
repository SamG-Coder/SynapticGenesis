"""Independent CPU forward oracle for answer scoring and generation (not training)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from associative_reference import forward as associative_forward

module = importlib.util.spec_from_file_location('lessons', Path(__file__).parents[1] / 'scripts/prepare_lessons.py')
lessons = importlib.util.module_from_spec(module)
module.loader.exec_module(lessons)
torch.set_num_threads(1)


class Reference:
    def __init__(self, checkpoint, association_mode='normal'):
        data = checkpoint.read_bytes()
        meta = struct.unpack_from('<32Q', data)
        self.cell, self.c, self.h, self.l = meta[1:5]
        if association_mode not in ('normal', 'discard_history', 'zero_read') or (self.cell != 6 and association_mode != 'normal'):
            raise ValueError('Invalid fast-memory intervention for this architecture')
        self.association_mode = association_mode
        w = torch.tensor(np.frombuffer(data, dtype='<f4', count=meta[14], offset=288).copy())
        offset = 0

        def take(*shape):
            nonlocal offset
            count = int(np.prod(shape))
            part = w[offset:offset + count].reshape(shape)
            offset += count
            return part

        self.embedding = take(256, self.c)
        self.blocks = []
        for _ in range(self.l):
            block = [take(self.c), take(self.h, self.c), take(self.h), take(self.c, self.h),
                     take(self.c), take(self.h)]
            if self.cell in (2, 3, 4, 5, 6):
                block += [take(self.h), take(self.h)]
            if self.cell in (4, 5, 6):
                block += [take(self.h, self.c), take(self.h)]
            if self.cell == 6:
                block += [take(98, self.h), take(98), take(self.c, 32), take(self.c)]
            self.blocks.append(block)
        self.gain, self.head, self.bias = take(self.c), take(256, self.c), take(256)
        assert offset == len(w)

    def logits(self, text, traces=None, forced_spikes=None):
        x = self.embedding[torch.tensor(list(text), dtype=torch.long)]

        def norm(x, gain):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * gain

        for layer, block in enumerate(self.blocks):
            gain, wi, bi, wo, bo, leak = block[:6]
            normalized = norm(x, gain)
            z = F.linear(normalized, wi, bi)
            gate_logits = F.linear(normalized, block[8], block[9]) if self.cell in (4, 5, 6) else None
            beta, reset, adaptation = torch.sigmoid(leak), torch.zeros(self.h), torch.zeros(self.h)
            if self.cell in (2, 3, 4, 5, 6):
                rho, gamma = torch.sigmoid(block[6]), F.softplus(block[7])
            spikes, membranes, decisions = [], [], []
            for at in range(len(text)):
                u = beta * reset + z[at]
                theta = 1 + gamma * adaptation if self.cell == 2 else 1
                spike = (u >= theta).float() - (u <= -theta).float()
                if forced_spikes is not None:
                    # Diagnostic intervention only: hold native spike choices
                    # fixed to isolate continuous arithmetic from threshold flips.
                    override = forced_spikes[layer][at]
                    spike = torch.where(torch.isnan(override), spike, override)
                if traces is not None:
                    membranes.append(u)
                    decisions.append(spike)
                reset = u - theta * spike
                if self.cell == 2:
                    adaptation = rho * adaptation + (1 - rho) * spike.abs()
                if self.cell in (3, 4, 5, 6):
                    retention = torch.sigmoid(block[6] + gate_logits[at]) if self.cell in (5, 6) else rho
                    adaptation = retention * adaptation + (1 - retention) * spike
                read = 2 * torch.sigmoid(gate_logits[at]) if self.cell == 4 else 1
                spikes.append(spike + gamma * read * adaptation if self.cell in (3, 4, 5, 6) else spike)
            emission = torch.stack(spikes)
            if traces is not None:
                traces.append(dict(norm=normalized, z=z, gate=gate_logits, u=torch.stack(membranes),
                                   spikes=torch.stack(decisions), emission=emission))
            x += F.linear(emission, wo, bo)
            if self.cell == 6:
                output, _ = associative_forward(emission.unsqueeze(0), *block[10:14], mode=self.association_mode)
                x += output.squeeze(0)
        return F.linear(norm(x, self.gain), self.head, self.bias)

    def score(self, prompt, answer):
        p, a = prompt.encode('ascii'), answer.encode('ascii')
        text = p + a
        logp = self.logits(text[:-1]).log_softmax(-1)
        return -sum(float(logp[len(p) - 1 + j, byte]) for j, byte in enumerate(a))

    def greedy(self, prompt, length):
        text = prompt.encode('ascii')
        answer = bytearray()
        for _ in range(length):
            byte = int(self.logits(text).argmax(-1)[-1])
            text += bytes([byte])
            answer.append(byte)
        return answer.decode('latin1')


def check(exe, out):
    exe, out = exe.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    calls = 0

    def run(*args, reject=None):
        nonlocal calls
        calls += 1
        r = subprocess.run([str(exe), *map(str, args)], capture_output=True)
        (out / f'command-{calls:02d}.log').write_bytes(r.stdout + r.stderr)
        if reject:
            assert r.returncode and reject.encode() in r.stderr, (args, r.stdout, r.stderr)
        else:
            assert r.returncode == 0, (args, r.stdout, r.stderr)

    rows = []
    for pair, query, choices in [('short', '?', ['X', 'Y']),
                                 ('quoted', '\nWhere?\nAnswer: ', ['box.', 'a bag.'])]:
        for gold, fact in enumerate(['A says "red".\\\n', 'A says "blue".\\\n']):
            rows.append(dict(id=f'{pair}-{gold}', pair=pair, skill=pair, correct=gold,
                             context=fact, query=query, choice0=choices[0], choice1=choices[1]))
    suite = out / 'probes.sgprobe'
    lessons.write_probes(suite, rows)
    (out / 'data.dat').write_bytes(b'The key is in the box.\nWhere is the key?\nAnswer: box.\n')
    max_error = 0
    for cell in ('lif', 'alif', 'trace', 'gated', 'selective', 'associative'):
        model = out / cell
        run('live', '--data', out / 'data.dat', '--out', model, '--channels', 8, '--hidden', 16,
            '--layers', 2, '--cell', cell, '--chunk', 8, '--updates', 4, '--speak-every', 0, '--lr', .001)
        checkpoint = model / 'latest.ckpt'
        initial_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        output = model / 'probes.json'
        run('language-probes', '--checkpoint', checkpoint, '--probes', suite, '--output', output)
        ref = Reference(checkpoint)
        report = json.loads(output.read_text())
        assert report['context_erased_paired_accuracy'] == 0 and report['parameters_and_optimizer_unchanged']
        assert report['strict_fp32'] and report['pairs'] == 2 and report['items'] == 4
        nll, targets = 0, 0
        for row, actual in zip(rows, report['results']):
            for erased, field in [(False, 'candidate_nll'), (True, 'context_erased_nll')]:
                prompt = row['query'] if erased else row['context'] + row['query']
                expected = [ref.score(prompt, row[f'choice{j}']) for j in range(2)]
                error = max(abs(a - b) for a, b in zip(expected, actual[field]))
                max_error = max(max_error, error)
                assert error < 3e-5, (cell, row['id'], field, expected, actual[field])
            assert actual['greedy'] == ref.greedy(row['context'] + row['query'],
                                                 max(len(row['choice0']), len(row['choice1'])))
            nll += actual['candidate_nll'][row['correct']]
            targets += len(row[f"choice{row['correct']}"])
        assert abs(report['answer_loss_nats_per_byte'] - nll / targets) < 1e-9
        assert report['answer_bytes'] == targets
        reverse = out / f'{cell}-reverse.sgprobe'
        lessons.write_probes(reverse, list(reversed(rows)))
        run('language-probes', '--checkpoint', checkpoint, '--probes', reverse,
            '--output', model / 'reordered.json')
        reordered = json.loads((model / 'reordered.json').read_text())
        assert report['results'] == list(reversed(reordered['results']))
        assert initial_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        run('language-probes', '--checkpoint', checkpoint, '--probes', suite, '--output', output,
            reject='Probe output exists')
        for kind in ('same-context', 'same-label', 'duplicate-id', 'bad-pair', 'trailing', 'truncated'):
            modified = [dict(x) for x in rows]
            if kind == 'same-context':
                modified[1]['context'] = modified[0]['context']
            elif kind == 'same-label':
                modified[1]['correct'] = modified[0]['correct']
            elif kind == 'duplicate-id':
                modified[1]['id'] = modified[0]['id']
            elif kind == 'bad-pair':
                modified[1]['query'] += 'x'
            bad = out / f'{cell}-{kind}.sgprobe'
            lessons.write_probes(bad, modified)
            if kind == 'trailing':
                bad.write_bytes(bad.read_bytes() + b'extra')
            if kind == 'truncated':
                bad.write_bytes(bad.read_bytes()[:-8])
            dest = out / f'{cell}-{kind}.json'
            run('language-probes', '--checkpoint', checkpoint, '--probes', bad, '--output', dest,
                reject='ERROR:')
            assert not dest.exists()
        if cell in ('trace', 'gated', 'selective', 'associative'):
            for options in (['--spike-add'], ['--spike-add', '--graph']):
                run('sample', '--checkpoint', checkpoint, '--tokens', 2, *options,
                    reject='Trace cell output is not ternary')
            paging = model / 'unsupported-paging'
            run('storage-bench', '--checkpoint', checkpoint, '--data', out / 'data.dat', '--out', paging,
                reject='Indexed spike paging does not support filtered trace emissions')
            assert not paging.exists()
            assert initial_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Reproducibility and split isolation are evaluated without a model or any
    # held-out answers being presented to a native learner.
    spec = Path(__file__).parents[1] / 'data/lessons-relations-v1.json'
    lessons.prepare(spec, out / 'prepared-a')
    lessons.prepare(spec, out / 'prepared-b')
    for p in (out / 'prepared-a').iterdir():
        assert p.read_bytes() == (out / 'prepared-b' / p.name).read_bytes()
    reading = out / 'prior-reading.dat'
    reading.write_bytes(b'A child sees a bird.\x1ePlants need water.\n')
    lessons.prepare(spec, out / 'with-reading', reading)
    assert (out / 'with-reading/reading.dat').read_bytes() == reading.read_bytes()
    assert (out / 'with-reading/reading-lessons.dat').read_bytes() == (
        reading.read_bytes() + b'\x1e' + (out / 'prepared-a/train.dat').read_bytes())
    for name, newline in [('lf', b'\n'), ('crlf', b'\r\n')]:
        contaminated = out / f'contaminated-{name}.dat'
        contaminated.write_bytes(b'The key is in the box. The hat is in the bag.' + newline)
        rejected_out = out / f'contaminated-{name}-output'
        try:
            lessons.prepare(spec, rejected_out, contaminated)
        except ValueError as error:
            assert 'held-out' in str(error)
        else:
            raise AssertionError('Contaminated prior reading was accepted')
        assert not rejected_out.exists()
    result = dict(passed=True, native_commands=calls, cells=['lif', 'alif', 'trace', 'gated', 'selective', 'associative'],
                  independent_cpu_forward_max_score_error=max_error,
                  independent_cpu_greedy_matches=True, answer_only_scoring_verified=True,
                  pair_order_independent=True, checkpoint_files_unchanged=True,
                  malformed_pairs_rejected=True, prepared_bytes_reproducible=True,
                  reading_bytes_preserved=True, heldout_reading_lf_and_crlf_rejected=True,
                  heldout_contexts_absent_from_lessons=True, synthetic_fixture_only=True,
                  incompatible_trace_spike_additions_and_paging_rejected=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.exe, args.out)
