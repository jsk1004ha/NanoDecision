"""Strict, versioned source parsing and unambiguous file selection.

Literal-quote TSV is opt-in. An explicit file variant never falls back to an
unrelated Parquet release. No source text is included in parse error messages.
"""
import csv
import fnmatch
import gzip
from pathlib import PurePosixPath
import re
import threading

SOURCE_CONTRACT_VERSION = 2
_FIELD_LIMIT_LOCK = threading.RLock()
_SUFFIXES = ('.parquet', '.jsonl', '.jsonl.gz', '.json', '.tsv', '.tsv.gz', '.csv', '.csv.gz')


class SourceContractError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f'{code}: {detail}')


def select_files(files, config, split, explicit=None, *, nested_explicit=False):
    """Select a single declared layout, never union different source variants."""
    if not isinstance(split, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', split):
        raise SourceContractError('invalid_split', 'Expected a simple split name')
    candidates = sorted(set(x for x in files if x.endswith(_SUFFIXES)))
    if explicit:
        hits = [x for x in candidates if any(fnmatch.fnmatchcase(x, p) for p in explicit)]
        if not hits and nested_explicit:
            # Only a literal basename can be relocated. Broad globs stay explicit.
            if any('/' in p or any(c in p for c in '*?[') for p in explicit):
                raise SourceContractError('invalid_nested_pattern', 'Use exact basenames for nested lookup')
            hits = [x for x in candidates if PurePosixPath(x).name in explicit]
            parents = {str(PurePosixPath(x).parent) for x in hits}
            if len(parents) > 1:
                raise SourceContractError('ambiguous_source', 'The explicit basename occurs in multiple directories')
        for x in hits:
            name = PurePosixPath(x).name
            if re.match(r'^(train|test|validation|dev)(?:[_.-]|$)', name):
                if not re.match(r'^' + re.escape(split) + r'(?:[_.-]|$)', name):
                    raise SourceContractError('split_mismatch', f'Explicit file does not belong to {split}')
        return hits

    def in_config(cfg):
        prefix = '' if cfg == '' else cfg + '/'
        layouts = [
            [f'{prefix}{split}/*.parquet'],
            [f'{prefix}{split}-*.parquet', f'{prefix}{split}.parquet'],
            [f'{prefix}{split}.jsonl', f'{prefix}{split}.jsonl.gz', f'{prefix}{split}.json'],
            [f'{prefix}data/{split}-*.parquet'],
        ]
        for patterns in layouts:
            hits = [x for x in candidates if any(fnmatch.fnmatchcase(x, p) for p in patterns)]
            if hits:
                return hits
        return []

    if config not in (None, 'default'):
        return in_config(config)
    # Allow only documented conventional layouts when the config is omitted.
    matches = {tuple(hit) for cfg in ('default', '', 'data', 'plain_text')
               if (hit := in_config(cfg))}
    if len(matches) > 1:
        raise SourceContractError('ambiguous_source', 'Multiple layouts match; set config/files explicitly')
    return list(next(iter(matches))) if matches else []


def metadata_files(card, files, config, split):
    """Read declared HF data_files metadata without executing dataset scripts."""
    if hasattr(card, 'to_dict'):
        card = card.to_dict()
    if not isinstance(card, dict):
        return []
    configs = card.get('configs', [])
    if not isinstance(configs, list):
        return []
    configs = [c for c in configs if isinstance(c, dict)]
    if config is not None:
        configs = [c for c in configs if c.get('config_name') == config]
    elif len(configs) > 1:
        defaults = [c for c in configs if c.get('default') is True]
        if len(defaults) != 1:
            raise SourceContractError('ambiguous_config', 'Specify the dataset configuration')
        configs = defaults
    if len(configs) != 1:
        return []
    entries = configs[0].get('data_files', [])
    if isinstance(entries, dict):
        entries = [{'split': k, 'path': v} for k, v in entries.items()]
    patterns = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get('split') == split:
            value = entry.get('path', [])
            patterns.extend([value] if isinstance(value, str) else value)
    if not all(isinstance(x, str) for x in patterns):
        raise SourceContractError('invalid_data_files', 'Metadata contains a non-string path')
    return select_files(files, config, split, patterns) if patterns else []


def delimited_rows(path, spec, start=0):
    """Yield logical record indexes; preserve quotes, empty fields and newlines."""
    if type(start) is not int or start < 0:
        raise SourceContractError('invalid_cursor', 'start must be a nonnegative record index')
    parser = spec.get('parser', {})
    if not isinstance(parser, dict):
        raise SourceContractError('invalid_parser', 'parser must be an object')
    quoting_name = parser.get('quoting', 'minimal')
    if quoting_name not in ('minimal', 'none'):
        raise SourceContractError('invalid_parser', 'quoting must be minimal or none')
    quoting = csv.QUOTE_NONE if quoting_name == 'none' else csv.QUOTE_MINIMAL
    limit = parser.get('field_size_limit', 4 * 1024 * 1024)
    if type(limit) is not int or not 1 <= limit <= 32 * 1024 * 1024:
        raise SourceContractError('invalid_parser', 'field_size_limit must be 1..32MiB characters')
    expected = parser.get('expected_columns')
    if expected is not None and (not isinstance(expected, list) or not expected
                                 or not all(isinstance(x, str) for x in expected)):
        raise SourceContractError('invalid_parser', 'expected_columns must be a nonempty list')
    opener = gzip.open if str(path).endswith('.gz') else open
    line_num = 0
    try:
        with opener(path, 'rt', encoding=parser.get('encoding', 'utf-8-sig'), newline='') as f:
            reader = csv.reader(
                f, delimiter=parser.get('delimiter', ',' if str(path).endswith(('.csv', '.csv.gz')) else '\t'), quoting=quoting,
                quotechar=None if quoting == csv.QUOTE_NONE else parser.get('quotechar', '"'),
                escapechar=parser.get('escapechar'), doublequote=parser.get('doublequote', True),
                strict=True,
            )
            def next_record():
                # csv.field_size_limit is process-global. Restore it between reads.
                with _FIELD_LIMIT_LOCK:
                    old = csv.field_size_limit()
                    try:
                        csv.field_size_limit(limit)
                        return next(reader)
                    finally:
                        csv.field_size_limit(old)
            try:
                columns = next_record()
            except StopIteration:
                raise SourceContractError('empty_source', 'Missing header') from None
            line_num = reader.line_num
            if not columns or any(not x for x in columns) or len(set(columns)) != len(columns):
                raise SourceContractError('invalid_header', 'Empty or duplicate column names')
            if expected is not None and columns != expected:
                raise SourceContractError('header_mismatch', 'Header does not match declared columns')
            index = 0
            while True:
                try:
                    row = next_record()
                except StopIteration:
                    break
                line_num = reader.line_num
                if len(row) != len(columns):
                    raise SourceContractError('column_count', f'record={index}, physical_line={line_num}, expected={len(columns)}, actual={len(row)}')
                if index >= start:
                    yield index, dict(zip(columns, row)), None
                index += 1
    except (csv.Error, UnicodeError) as exc:
        raise SourceContractError('parse_error', f'{type(exc).__name__} near physical_line={line_num}; source={spec.get("id", "unknown")}') from None
