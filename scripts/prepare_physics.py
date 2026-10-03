"""Prepare the admitted Physics edition with whole-module overlap protection."""
import argparse
from pathlib import Path
import re

from corpus.paired_selection import select_documents
from corpus.physics_edits import digest, require
from corpus.selection import require_training_spec
from native_experiment import read
from prepare_corpus import fetch
from prepare_openstax import corrected_body
from prepare_prealgebra import protected_inputs
from prose_founder import write
from review_physics import authenticate
from review_physics_edition import compile_edition


def reviewed_modules(spec, cache, download=False):
    for key in ('source_review_spec', 'source_edits', 'passage_review', 'corrected_review'):
        require(digest(Path(spec[key]).read_bytes()) == spec[key + '_sha256'], 'Review input changed: ' + key)
    passage = read(spec['passage_review'])
    review = read(spec['corrected_review'])
    require(passage['complete'] and passage['source_review_sha256'] == spec['corrected_review_sha256'],
            'Passage review refers to a different extraction')
    upstream, edits, auxiliary, bodies, records = compile_edition(
        Path(spec['source_review_spec']), Path(spec['source_edits']), cache, download)
    require(records == review['records'] and upstream['upstream_commit'] == spec['upstream_commit'] and
            upstream['upstream_repository'] == spec['upstream_repository'], 'Reviewed Physics source differs')
    for image in passage['additional_inspected_media']:
        name = image['source_url'].rsplit('/',1)[-1]
        require(re.fullmatch(r'[A-Za-z0-9_.-]+\.jpg', name) and image['source_url'] ==
            f"https://raw.githubusercontent.com/openstax/osbooks-physics/{spec['upstream_commit']}/media/{name}",
            'Unexpected inspected diagram URL')
        path = cache / 'raw/physics-media' / name
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            fetch(image['source_url'], path)
        authenticate(path.read_bytes(), image)
    samples = passage['reviewed_samples']
    require(len(samples) == 30 and len({s['module'] for s in samples}) == 30, 'Passage sample set changed')
    identities = {r['module']: r for r in records}
    for sample in samples:
        require(sample['module'] in bodies and digest(bodies[sample['module']]) == sample['text_sha256'] and
                sample['sample'].strip() and sample['sample'] in bodies[sample['module']].decode('utf-8'),
                'Reviewed passage changed')
        original = identities[sample['module']]
        require(all(sample[key] == original[key] for key in ('title', 'chapter')), 'Reviewed identity changed')
        corrections = [r['id'] for r in passage['text_corrections'] if r['module'] == sample['module']]
        decision = 'retain_with_explicit_correction' if corrections else 'retain_at_introductory_textbook_level'
        require(sample['correction_ids'] == corrections and sample['decision'] == decision,
                'Passage decision does not match its corrections')
    require(len(passage['text_corrections']) == 5 and
            len({r['id'] for r in passage['text_corrections']}) == 5, 'Correction set changed')
    chapter_names = list(dict.fromkeys(r['chapter'] for r in records if r['chapter'] != 'Front or back matter'))
    require(len(chapter_names) == 23 and set(spec['chapter_policy']) == set(chapter_names + ['Front or back matter']),
            'Chapter policy does not cover the source collection')
    require(set(chapter_names).issubset({s['chapter'] for s in samples}), 'Chapter missing from passage review')
    require([s['id'] for s in spec['stages']] == [1, 2, 3, 4] and all(s['name'].strip() for s in spec['stages']),
            'Topic stages must occur once in curriculum order')
    for number, name in enumerate(chapter_names, 1):
        policy = spec['chapter_policy'][name]
        expected_split = 'validation' if number == 8 else 'test' if number == 19 else 'train'
        expected_stage = 1 if number <= 5 else 2 if number <= 12 else 3 if number <= 20 else 4
        require(policy == dict(chapter_number=number, split=expected_split, stage=expected_stage),
                'Protected chapter assignment changed')
    require(spec['chapter_policy']['Front or back matter'] ==
            dict(chapter_number=0, split='train', stage=4), 'Appendix policy changed')
    rows = []
    for record in records:
        ident = record['module']
        policy = spec['chapter_policy'][record['chapter']]
        require(policy['stage'] in (1,2,3,4), 'Unknown developmental topic stage')
        corrections = [r for r in passage['text_corrections'] if r['module'] == ident]
        text = bodies[ident].decode('utf-8')
        for correction in corrections:
            require(correction['reason'].strip() and correction['references'] and
                    correction['before'] and text.count(correction['before']) == 1 and
                    correction['after'].strip() and correction['before'] != correction['after'],
                    'Ambiguous text correction')
            text = text.replace(correction['before'], correction['after'], 1)
        expected = dict(extracted_sha256=record['text_sha256'], corrections=corrections,
                        corrected_sha256=digest(text.encode('utf-8')))
        body = corrected_body(bodies[ident], expected)
        require(b'\x1e' not in body, 'Source contains a native document separator')
        rows.append(dict(id=ident, title=record['title'], chapter=record['chapter'], **policy,
            raw_sha256=record['raw_sha256'], extracted_sha256=record['text_sha256'],
            corrected_sha256=digest(body), correction_ids=[r['id'] for r in corrections], text=body.decode('utf-8')))
    require(sum(len(r['correction_ids']) for r in rows) == 5, 'An inspected correction was not applied')
    return rows, auxiliary, edits, passage


def compiled_selection(spec, cache, download=False):
    require(spec['version'] == 'selected-physics-v1', 'Unknown Physics edition')
    rows, auxiliary, edits, passage = reviewed_modules(spec, cache, download)
    selected, omitted = select_documents(rows, protected_inputs(spec['protected_files']))
    require(omitted == spec['overlap_omissions'], 'Whole-module overlap decisions changed')
    expected = {r['id']:r for r in spec['sources']}
    require(len(expected) == len(spec['sources']) == len(selected), 'Selected module count changed')
    for row in selected:
        require({k:v for k,v in row.items() if k != 'text'} == expected.get(row['id']), 'Selected module changed')
    return {r['id']:r['text'].encode('utf-8') for r in selected}, auxiliary, edits, passage


def prepare(spec_path, cache, out, download=False):
    require(not out.exists(), 'Use a fresh Physics prepared directory')
    spec = require_training_spec(spec_path)
    bodies, auxiliary, edits, passage = compiled_selection(spec, cache, download)
    outputs, stages = {}, []
    payloads = {f'{ident}.txt':body for ident,body in bodies.items()}
    for split in ('train','validation','test'):
        selected = [r for r in spec['sources'] if r['split'] == split]
        raw = b'\x1e'.join(bodies[r['id']] for r in selected)
        require(bool(raw), 'Every split must contain a complete module')
        payloads[split + '.dat'] = raw
        outputs[split] = dict(bytes=len(raw), documents=len(selected), sha256=digest(raw),
            word_like_units=len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*",raw.decode('utf-8'))))
    for stage in spec['stages']:
        raw = b'\x1e'.join(bodies[r['id']] for r in spec['sources'] if r['split']=='train' and r['stage']==stage['id'])
        require(bool(raw), 'Empty topic stage')
        name = f"stage-{stage['id']}.dat"
        payloads[name] = raw
        stages.append(dict(stage,file=name,bytes=len(raw),sha256=digest(raw)))
    manifest = dict(version=spec['version'],source_spec_sha256=digest(spec_path.read_bytes()),
        source_repository=spec['upstream_repository'],source_commit=spec['upstream_commit'],
        provider='reviewed-openstax-physics',tokenizer='UTF-8 bytes, fixed IDs 0..255',
        document_separator='0x1e, excluded from sampled windows',license='CC BY 4.0',attribution=spec['attribution'],
        outputs=outputs,stages=stages,sources=[dict(r,clean_bytes=len(bodies[r['id']]),clean_sha256=digest(bodies[r['id']])) for r in spec['sources']],
        protected_files=spec['protected_files'],overlap_omissions=spec['overlap_omissions'],
        module_content_preserved_after_declared_edits=True,models_trained=False,curriculum_created=False,
        limitations=spec['limitations'])
    payloads.update({'source-spec.json':spec_path.read_bytes(),'passage-review.json':Path(spec['passage_review']).read_bytes(),
        'source-edits.json':Path(spec['source_edits']).read_bytes(),'LICENSE-source.txt':auxiliary['license'],
        'original-preface.cnxml':auxiliary['preface'],'original-collection.xml':auxiliary['collection'],
        'ATTRIBUTION.txt':(spec['attribution']+'\n').encode('utf-8')})
    out.mkdir(parents=True)
    for name,raw in payloads.items(): (out/name).write_bytes(raw)
    write(out/'manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',type=Path,default=Path('data/sources-physics-v1.json'))
    parser.add_argument('--cache',type=Path,default=Path('runs/science-source-review'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    result=prepare(args.sources,args.cache,args.out,args.download)
    print(result['outputs'])
