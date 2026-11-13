"""Resolve and probe public sources before downloading the encoder weights."""
from pathlib import Path
import argparse
import json
from itertools import islice
from .auto_sources import PublicReader, raw_rows, convert, atomic_json
from .source_contracts import SOURCE_CONTRACT_VERSION
from .schema import normalize_question, gold_target


def preflight(recipe, workdir, sample_records=3):
    if type(sample_records) is not int or not 1 <= sample_records <= 1000:
        raise ValueError('sample_records must be 1..1000')
    root = Path(workdir)
    reader = PublicReader(root / 'downloads')
    report = {'source_contract_version': SOURCE_CONTRACT_VERSION, 'sources': [],
              'scope': 'File resolution and first-shard sample validation, not a full-corpus audit'}
    errors = []
    for spec in recipe['sources']:
        if spec['adapter'] == 'nli_entails' and not recipe.get('derived_nli_entailment', False):
            continue
        result = {'id': spec['id'], 'required': spec.get('required', True)}
        try:
            result['resolved'] = reader.resolve(spec)
            shard = next(reader.shards(spec), None)
            if shard is None:
                raise ValueError('Source has no shards')
            shard_id, path, provenance = shard
            reading = dict(spec)
            if provenance.get('file', '').endswith('.parquet'):
                reading['_parquet'] = True
            rows = raw_rows(path, reading)
            seen = decisions = 0
            try:
                for index, raw, names in islice(rows, sample_records):
                    seen += 1
                    for decision in convert(spec, raw, f'{shard_id}:{index}', names):
                        gold_target(decision, normalize_question(decision))
                        decisions += 1
            finally:
                rows.close()
            if not seen:
                raise ValueError('No records in source sample')
            result.update(status='ok', sampled_records=seen,
                          converted_decisions=decisions, shard=provenance)
        except Exception as exc:
            result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            if result['required']:
                errors.append(spec['id'])
        report['sources'].append(result)
        report['required_failures'] = list(errors)
        atomic_json(root / 'source_preflight.json', report)
    if errors:
        raise RuntimeError('Required source preflight failed: ' + ', '.join(errors) +
                           '. See source_preflight.json; model download has not started.')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', required=True)
    p.add_argument('--workdir', required=True)
    p.add_argument('--samples', type=int, default=3)
    a = p.parse_args(argv)
    print(json.dumps(preflight(json.loads(Path(a.recipe).read_text()), a.workdir, a.samples),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
