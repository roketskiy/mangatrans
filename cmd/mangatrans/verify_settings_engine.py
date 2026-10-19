"""Drive the normal engine CLI; never build or execute Go test executables."""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = args.out / time.strftime('run-%Y%m%d-%H%M%S')
    run_dir.mkdir()
    report = {'passed': False, 'checks': [], 'artifacts': [], 'error': None}
    report_path = run_dir / 'engine-report.json'
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()

    def run(name, config, extra=(), expected=None, source=None, output=None):
        config.setdefault('version', 2)
        destination = output or run_dir / (name + '.cbz')
        command = [str(args.engine.resolve()), '-in', str((source or args.input).resolve()),
                   '-out', str(destination.resolve()), '-limit', '1',
                   '-settings-json', json.dumps(config, separators=(',', ':')),
                   *extra]
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                   errors='replace', timeout=args.timeout,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        log = completed.stdout + completed.stderr
        (run_dir / (name + '.log')).write_text(log, encoding='utf-8')
        if expected:
            assert completed.returncode != 0 and expected in log, (name, log)
            assert 'starting OCR server' not in log, (name, log)
        else:
            assert completed.returncode == 0, (name, log)
        return destination, log

    def inspect(path, extension):
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            assert len(entries) == 1
            entry = entries[0]
            assert entry.filename.endswith(extension), entry.filename
            with Image.open(io.BytesIO(archive.read(entry))) as image:
                image.load()
                assert max(image.size) <= 640, image.size
                report['artifacts'].append({'path': str(path), 'entry': entry.filename,
                                            'format': image.format, 'size': image.size})

    try:
        run('conflict', {}, ('-settings', 'missing.json'), expected='mutually exclusive')
        run('invalid-tiles', {'recognition': {'tile_size': 128}},
            expected='tile')
        run('invalid-confidence', {}, ('-conf', 'NaN'), expected='confidence')
        run('invalid-batch', {}, ('-mt-batch-pages', '0'), expected='batch pages')
        run('missing-font', {'layout': {'font_path': str(run_dir / 'missing.ttf')}},
            expected='required file')
        run('missing-model', {'translation': {'model_path': str(run_dir / 'missing.gguf')}},
            expected='required file')
        report['checks'].append('CLI source conflict, tile size, confidence, batch, font, and model guards')

        base = {
            'recognition': {'tile_size': 1024, 'confidence': 0.35},
            'repair': {'tile_size': 384, 'context_padding': 64, 'mask_dilation': 2},
            'layout': {'stroke_width': 0.75, 'spacing': 1.2, 'font_scale': 0.9, 'color': '#102030'},
            'output': {'format': 'png', 'max_long_edge': 640, 'preserve_no_text': False,
                       'save_editable': True, 'save_diagnostics': True, 'existing': 'rename'},
        }
        destination, first_log = run('first', base)
        assert 'starting OCR server' in first_log
        assert 'render failed' not in first_log and 'inpaint failed' not in first_log
        inspect(destination, '.png')
        editable = json.loads(Path(str(destination) + '.editable.json').read_text(encoding='utf-8'))
        assert editable['inference']['pages'][0]['ocr']
        assert all(block['Text'] for block in editable['inference']['pages'][0]['blocks'])
        assert list(Path(str(destination) + '.work').glob('run-*/mask00.png'))
        report['checks'].append('real 1024px tiled detection, OCR, translation, 384px repair cores, PNG, editable JSON, diagnostics')

        style = copy.deepcopy(base)
        style['repair'].update(mask_dilation=10, tile_size=768, context_padding=96)
        style['layout'].update(direction='horizontal', stroke_width=2.25, spacing=1.4,
                               font_scale=1.1, min_font_size=2, color='#123456')
        style['output'].update(format='webp', webp_lossless=True, save_diagnostics=False)
        _, log = run('style-lossless', style, output=destination)
        lossless = run_dir / 'first (2).cbz'
        assert 'reused recognition and translation' in log
        assert 'starting OCR server' not in log and 'starting translation server' not in log
        inspect(lossless, '.webp')
        assert not Path(str(lossless) + '.work').exists()
        report['checks'].append('style/repair/output changes reuse inference; rename protects prior output; lossless WebP')

        for name, fmt, extension in [('lossy', 'webp', '.webp'), ('jpeg', 'jpeg', '.jpg')]:
            changed = copy.deepcopy(style)
            changed['output'].update(format=fmt, webp_lossless=False, jpeg_quality=87, webp_quality=75)
            _, log = run(name, changed, output=destination)
            assert 'reused recognition and translation' in log
            number = 3 if name == 'lossy' else 4
            inspect(run_dir / f'first ({number}).cbz', extension)
        report['checks'].append('lossy WebP and JPEG quality settings produce matching codec names')

        changed = copy.deepcopy(base)
        changed['processing'] = {'reuse_results': False}
        changed['translation'] = {'batch_attempts': 2}
        changed['output']['save_diagnostics'] = False
        _, log = run('no-reuse', changed, output=destination)
        assert 'starting OCR server' in log and 'reused recognition' not in log
        inspect(run_dir / 'first (5).cbz', '.png')
        report['checks'].append('disabled reuse starts fresh OCR and translation despite a matching cache')

        changed = copy.deepcopy(base)
        changed['recognition']['confidence'] = 0.45
        changed['translation'] = {'batch_attempts': 5}
        changed['output']['save_diagnostics'] = False
        _, log = run('inference-change', changed, output=destination)
        assert 'starting OCR server' in log and 'reused recognition' not in log
        inspect(run_dir / 'first (6).cbz', '.png')
        report['checks'].append('numeric confidence and batch attempt changes invalidate inference cache')

        original_destination = run_dir / 'first-original.cbz'
        original_destination.write_bytes(destination.read_bytes())
        for artifact in report['artifacts']:
            if artifact['path'] == str(destination):
                artifact['path'] = str(original_destination)
        denied = copy.deepcopy(base)
        denied['output']['existing'] = 'ask'
        run('existing-denied', denied, output=destination, expected='output exists')
        assert destination.read_bytes() == original_destination.read_bytes()
        confirmed = copy.deepcopy(style)
        confirmed['output']['existing'] = 'ask'
        _, log = run('confirmed-overwrite', confirmed, extra=('-overwrite',), output=destination)
        assert 'reused recognition and translation' in log
        inspect(destination, '.webp')
        assert destination.read_bytes() != original_destination.read_bytes()
        assert hashlib.sha256(args.input.read_bytes()).hexdigest() == source_hash
        report['checks'].append('ask rejects existing output without changes; explicit overwrite replaces only confirmed output')

        source = run_dir / 'blank-source.cbz'
        originals = []
        for name, fmt in [('1.png', 'PNG'), ('1.webp', 'WEBP')]:
            data = io.BytesIO()
            Image.new('RGB', (800, 1000), 'white').save(data, format=fmt)
            originals.append((name, data.getvalue()))
        originals.append(('10.png', b'intentionally undecodable original page'))
        with zipfile.ZipFile(source, 'w') as archive:
            for name, data in originals:
                archive.writestr(name, data)
        for preserve in [True, False]:
            name = 'blank-preserve' if preserve else 'blank-convert'
            output, _ = run(name, {'output': {'format': 'png', 'max_long_edge': 640,
                                              'preserve_no_text': preserve}},
                            extra=('-limit', '0'), source=source)
            with zipfile.ZipFile(output) as archive:
                entries = archive.infolist()
                assert len(entries) == 3 and len({e.filename for e in entries}) == 3
                assert [e.filename[:8] for e in entries] == ['00000001', '00000002', '00000003']
                for index, entry in enumerate(entries):
                    data = archive.read(entry)
                    if preserve or index == 2:
                        assert data == originals[index][1]
                    else:
                        assert entry.filename.endswith('.png')
                        with Image.open(io.BytesIO(data)) as image:
                            assert max(image.size) == 640
            report['artifacts'].append({'path': str(output), 'entries': [e.filename for e in entries]})
        report['checks'].append('blank PNG/WebP preservation and conversion; failed page bytes unchanged; unique natural order')
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(report_path)
    print('PASS: normal engine CLI verification; no Go test executables')


if __name__ == '__main__':
    main()
