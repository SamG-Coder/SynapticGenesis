"""Independent host selection/exposure audit and learned before/after CPU scores."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_checkpoint import checkpoint
from extend_curriculum import read_schedule
from native_experiment import read, sha, write
from replay_priority_probe import cpu_scores, journal, speech_bytes
from stage_replay_reference import Group, MASK, Random64, ReplayReference


def restore_reference(path, docs, first):
    # Reuse the existing independent policy stepping with an unchanged stage.
    meta, extra, *_ = checkpoint(path)
    ref = ReplayReference.__new__(ReplayReference)
    ref.meta, ref.header = list(meta), list(extra[:16])
    ref.random, ref.speech = Random64(extra[4]), Random64(meta[23])
    ref.first, ref.lengths, ref.groups = first, [len(d) for d in docs], []
    start = 17 + 5 * extra[16]
    for index in range(extra[16]):
        end, seen, count, updates, pairs = extra[17 + index * 5:22 + index * 5]
        items = [tuple(extra[j:j + 3]) for j in range(start, start + count * 3, 3)]
        ref.groups.append(Group(end, seen, updates, pairs, items))
        start += count * 3
    assert start == len(extra)
    return ref


def expected_candidates(ref, seed, per_group):
    rng = Random64((seed ^ (((ref.meta[24] + 1) * 0x9e3779b97f4a7c15) & MASK)) or 1)
    expected, start = [], 0
    for index, group in enumerate(ref.groups):
        remaining = list(range(len(group.items)))
        for _ in range(min(len(remaining), per_group)):
            chosen = rng.below(len(remaining))
            slot = remaining[chosen]
            doc, offset, length = group.items[slot]
            expected.append(dict(slot=start + slot, group=index + 1, document=doc, offset=offset, length=length))
            remaining[chosen] = remaining[-1]
            remaining.pop()
        start += len(group.items)
    return expected


def audit(root, out):
    comparison = read(root / 'comparison.json')
    protocol = comparison['protocol']
    assert comparison['complete'] and comparison['inputs_unchanged']
    assert not protocol['reserved_tests_scored'] and not protocol['priority_replay_enabled']
    assert all(sha(p) == identity for p, identity in protocol['authenticated_inputs'].items())
    assert all(sha(p) == identity for p, identity in protocol['source_sha256'].items())
    _, stages = read_schedule(Path(protocol['schedule']))
    docs = stages[-1]['content'].split(b'\x1e')
    first = len(stages[-2]['content'].split(b'\x1e'))
    answer_documents, start = set(), 0
    for stage in stages:
        end = len(stage['content'].split(b'\x1e'))
        if float(stage['answer']) > 1:
            assert float(stage['answer']) == 64
            answer_documents.update(range(start, end))
        start = end
    results = []
    for base in protocol['bases']:
        seed = base['seed']
        rows = [r for r in comparison['runs'] if r['seed'] == seed]
        expected = root / f'{seed}-original-control/latest.ckpt'
        generated = speech_bytes(root / f'{seed}-original-control/transcript.txt')
        assert generated == speech_bytes(root / f'{seed}-native-control/transcript.txt')
        assert sha(expected) == sha(root / f'{seed}-native-control/latest.ckpt')
        records = journal(root / f'{seed}-0-measured/scores.jsonl')
        assert [r['source_observation'] for r in records] == list(range(160001, 160513, 32))
        ref = restore_reference(Path(base['checkpoint']), docs, first)
        for event in records:
            ref.run_until(event['source_observation'] - 1)
            doc, offset = ref.meta[19:21]
            assert event['global_update'] == ref.meta[7] + 1
            assert event['source'] == dict(document=doc, offset=offset,
                                           length=min(ref.meta[6], len(docs[doc]) - 1 - offset))
            candidates = expected_candidates(ref, protocol['candidate_seed'], protocol['candidates_per_group'])
            assert candidates == [{k: c[k] for k in candidates[0]} for c in event['candidates']]
        ref.run_until(protocol['end'])
        policy = ref.matches(expected)
        for row in rows:
            directory = root / row['directory']
            assert sha(directory / 'latest.ckpt') == sha(expected)
            assert (directory / 'speech.txt').read_bytes() == generated
            assert sha(directory / 'scores.jsonl') == row['scores_sha256']
            scores = journal(directory / 'scores.jsonl')
            assert scores == (records if row['mode'] == 'measured' else [])
            assert row['result'] == read(directory / 'result.json')
            expected_calls = 2 * sum(len(r['candidates']) for r in scores)
            expected_pairs = 2 * sum(c['length'] for r in scores for c in r['candidates'])
            assert row['result']['score_forward_calls'] == expected_calls
            assert row['result']['scored_pairs'] == expected_pairs
        oracle_dir = root / f'{seed}-oracle'
        assert journal(oracle_dir / 'scores.jsonl')[0] == records[0]
        oracle = cpu_scores(oracle_dir / 'score-before.ckpt', oracle_dir / 'score-after.ckpt',
                            docs, records[0]['candidates'], answer_documents)
        write(oracle_dir / 'cpu-oracle.json', oracle)
        results.append(dict(seed=seed, independent_policy=policy, candidate_pools_exact=True,
                            exact_checkpoint_and_speech=True, cpu_oracle=oracle))
        print(seed, 'execution exact; CPU score pass:', oracle['passed'], 'error:', oracle['max_abs_error'], flush=True)
    result = dict(execution_passed=True, all_cpu_scores_passed=all(r['cpu_oracle']['passed'] for r in results),
                  comparison_sha256=sha(root / 'comparison.json'), measured_observations=48,
                  candidate_scores_per_phase=1536, rows=results,
                  scope='CPU check covers the first 32 candidates before/after one update in each seed. '
                        'It does not establish parity at all 48 measured observations or learning quality.')
    write(out, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.out)
