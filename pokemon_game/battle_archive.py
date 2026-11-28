"""Combine compatible captured battle archives, preserving scenario provenance."""
import argparse
import hashlib
import json
from pathlib import Path


def merge(inputs, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new archive directory: {output}')
    entries, rom_hash = [], None
    for path in map(lambda p: Path(p).resolve(), inputs):
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest.get('version') != 1:
            raise ValueError('Unsupported battle archive version')
        if rom_hash is None:
            rom_hash = manifest['rom_sha256']
        if manifest['rom_sha256'] != rom_hash:
            raise ValueError('Cannot merge archives from different ROMs')
        for scenario in manifest['scenarios']:
            filename = scenario['snapshot']
            if Path(filename).name != filename:
                raise ValueError('Invalid snapshot filename')
            data = (path / filename).read_bytes()
            if hashlib.sha256(data).hexdigest() != scenario['snapshot_sha256']:
                raise ValueError('Snapshot hash mismatch')
            entries.append((scenario, data, path.name))
    if not entries:
        raise ValueError('No scenarios provided')
    output.mkdir(parents=True)
    scenarios = []
    for index, (scenario, data, source) in enumerate(entries, 1):
        filename = f'battle_{index:04d}.state'
        (output / filename).write_bytes(data)
        scenarios.append({**scenario, 'snapshot': filename, 'source_archive': source})
    (output / 'manifest.json').write_text(json.dumps({'version': 1, 'rom_sha256': rom_hash,
                                                       'scenarios': scenarios}, indent=2) + '\n')
    print(f'Merged {len(scenarios)} states into {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    merge(args.inputs, args.output)


if __name__ == '__main__':
    main()
