"""Run a bounded engine regression and verify that the stalled page was rendered."""
import argparse
import io
import json
import pathlib
import subprocess
import tempfile
import zipfile

from PIL import Image

from winjob import Job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=pathlib.Path)
    parser.add_argument('--settings', required=True, type=pathlib.Path)
    parser.add_argument('--engine', type=pathlib.Path, default=pathlib.Path('bin/mangatrans.exe'))
    parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('.cache/render-verification/e2e'))
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    root = pathlib.Path(__file__).resolve().parent.parent
    args.out.mkdir(parents=True, exist_ok=True)
    dest = pathlib.Path(tempfile.mkdtemp(prefix='run-', dir=args.out.resolve()))
    config = json.loads(args.settings.read_text(encoding='utf-8-sig'))
    config['output']['save_diagnostics'] = True
    config['processing']['reuse_results'] = False
    (dest / 'settings.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    command = [str(args.engine.resolve()), '-in', str(args.input.resolve()),
               '-out', str(dest / 'result.cbz'), '-limit', '2',
               '-settings', str(dest / 'settings.json'), '-work', str(dest / 'work'),
               '-perf-json', str(dest / 'perf.json')]
    report = {'command': command, 'passed': False, 'timeout_seconds': args.timeout}
    job = Job()
    try:
        with (dest / 'engine.log').open('wb') as log:
            proc = job.launch(command, log, str(root))
            try:
                report['exit_code'] = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                job.terminate()
                proc.wait(timeout=10)
                raise RuntimeError('Engine exceeded its deadline') from None
        if report['exit_code'] != 0:
            raise RuntimeError('Engine failed; see engine.log')
        perf = json.loads((dest / 'perf.json').read_text(encoding='utf-8'))
        pages = perf['pages']
        if len(pages) != 2:
            raise RuntimeError('Engine did not process exactly two pages')
        rendered = pages[1]['rendered']
        if pages[1]['outcome'] != 'rendered' or type(rendered) is not int or rendered < 1:
            raise RuntimeError('The stalled page was skipped or failed instead of rendered')
        if perf.get('error'):
            raise RuntimeError(perf['error'])
        with zipfile.ZipFile(dest / 'result.cbz') as archive:
            names = archive.namelist()
            if len(names) != 2 or archive.testzip() is not None:
                raise RuntimeError('Invalid two-page result archive')
            for index, name in enumerate(names):
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    image.load()
                    if min(image.size) <= 0:
                        raise RuntimeError('Empty output image')
                    if index == 1:
                        image.save(dest / 'page-2.png')
        report['pages'] = pages
        report['render_ms'] = [e['duration_ns'] / 1e6 for e in perf['events']
                               if e['kind'] == 'operation' and e['stage'] == 'render']
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
    finally:
        job.close()
        (dest / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'artifacts': str(dest), **report}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
