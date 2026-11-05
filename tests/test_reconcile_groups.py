"""Reconciliation must produce input that the cache's leakage audit accepts."""
import json

import pytest

from nanodecision2.adapters import partition_for
from nanodecision2.data import digest, prepare_cache
from scripts.reconcile_groups import reconcile


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def row(identifier, group, state, **extra):
    return dict(id=identifier, group_id=group, state=state, source_split='train',
                partition='train', type='noul', question=f'Question {identifier}?',
                target=True, label_origin='dataset_label', **extra)


def cache_reconciled(tmp_path, output):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'init.json').write_text('{"smoke_only": true}')
    (source / 'config.json').write_text('{"max_position_embeddings": 512}')
    return prepare_cache([output], tmp_path / 'cache', source, max_state=512, max_pair=256)


@pytest.mark.parametrize('states', [('Ａ', 'A'), ('  ﬁle  ', 'file'),
                                   ({'value': '①'}, {'value': '1'})])
def test_compatibility_equivalent_states_merge_before_cache(tmp_path, states):
    inputs, output = tmp_path / 'rows.jsonl', tmp_path / 'merged.jsonl'
    write_rows(inputs, [row('1', 'a', states[0]), row('2', 'b', states[1])])
    report = reconcile([inputs], output)
    merged = [json.loads(line) for line in output.read_text().splitlines()]
    assert report['groups'] == 1
    assert len({item['group_id'] for item in merged}) == 1
    assert 'NFKC' in report['scope']
    cache_reconciled(tmp_path, output)


@pytest.mark.parametrize('synthetic_first', [False, True])
def test_synthetic_labels_move_the_entire_repartitionable_group(tmp_path, synthetic_first):
    group = next(f'g{i}' for i in range(1000) if partition_for(f'g{i}', 'train') == 'conformal')
    human = row('human', group, 'human state')
    synthetic = row('teacher', group, 'teacher state')
    synthetic['label_origin'] = 'teacher_distribution_synthetic'
    inputs, output = tmp_path / 'rows.jsonl', tmp_path / 'merged.jsonl'
    write_rows(inputs, [synthetic, human] if synthetic_first else [human, synthetic])
    report = reconcile([inputs], output, repartition_training=True)
    merged = [json.loads(line) for line in output.read_text().splitlines()]
    assert {item['partition'] for item in merged} == {'validation'}
    assert report['synthetic_groups_moved_to_validation'] == 1
    cache_reconciled(tmp_path, output)


def test_fixed_conformal_group_with_synthetic_labels_is_rejected(tmp_path):
    human, synthetic = row('human', 'group', 'one'), row('teacher', 'group', 'two')
    for item in (human, synthetic):
        item['partition'] = 'conformal'
    synthetic['label_origin'] = 'teacher_distribution_synthetic'
    inputs, output = tmp_path / 'rows.jsonl', tmp_path / 'merged.jsonl'
    write_rows(inputs, [human, synthetic])
    with pytest.raises(ValueError, match='fixed conformal'):
        reconcile([inputs], output)
    assert not output.exists()
    assert not output.with_name(output.name + '.partial').exists()


def test_human_only_fixed_conformal_group_is_preserved(tmp_path):
    item = row('human', 'group', 'one')
    item['partition'] = 'conformal'
    inputs, output = tmp_path / 'rows.jsonl', tmp_path / 'merged.jsonl'
    write_rows(inputs, [item])
    reconcile([inputs], output)
    assert json.loads(output.read_text())['partition'] == 'conformal'
    cache_reconciled(tmp_path, output)


def test_nfkc_duplicates_cannot_silently_move_official_test(tmp_path):
    training, held_out = row('train', 'a', 'Ａ'), row('held-out', 'b', 'A')
    held_out.update(source_split='test', partition='test')
    inputs, output = tmp_path / 'rows.jsonl', tmp_path / 'merged.jsonl'
    write_rows(inputs, [training, held_out])
    with pytest.raises(ValueError, match='fixed partitions'):
        reconcile([inputs], output, repartition_training=True)
    assert not output.exists()


@pytest.mark.parametrize('reverse', [False, True])
def test_synthetic_policy_applies_after_transitive_group_merges(tmp_path, reverse):
    group = next(f'g{i}' for i in range(1000) if partition_for(f'g{i}', 'train') == 'conformal')
    items = [row('human', group, 'A'), row('bridge', 'z-linked', 'Ａ'),
             row('teacher', 'z-linked', 'another state')]
    items[-1]['label_origin'] = 'teacher_distribution_synthetic'
    inputs, output = tmp_path / 'rows.jsonl', tmp_path / 'merged.jsonl'
    write_rows(inputs, list(reversed(items)) if reverse else items)
    report = reconcile([inputs], output, repartition_training=True)
    merged = [json.loads(line) for line in output.read_text().splitlines()]
    assert report['groups'] == 1
    assert {(item['group_id'], item['partition']) for item in merged} == {(group, 'validation')}
    cache_reconciled(tmp_path, output)


def test_autodata_uses_the_same_compatibility_normalization(tmp_path):
    from nanodecision2.autodata import Store
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'init.json').write_text('{"smoke_only": true}')
    (source / 'config.json').write_text('{"max_position_embeddings": 512}')
    store = Store(tmp_path / 'pool', {'max_state_tokens': 512, 'max_pair_tokens': 256}, source)
    try:
        for item in [row('first', 'a', 'Ａ'), row('second', 'b', 'A')]:
            assert store.add(item, {'id': 'fixture', 'split': 'train'}) == 'accepted'
        assert store.find(digest('a')) == store.find(digest('b'))
        assert store.db.execute('SELECT count(*) FROM states').fetchone()[0] == 1
    finally:
        store.close()
