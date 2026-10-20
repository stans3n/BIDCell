"""Read-only /proc sampling for the bounded maximum-K validation driver.

The numerical worker records exact high-water marks after computational stages.
This separate observer also samples retained-array/output-writing lifetime and
the concurrent parent+worker RSS. Sampling is an observed lower bound, never a
claim to the unsampled lifetime maximum.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def status(pid):
    try:
        fields = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmHWM'):
                fields[key] = int(value.split()[0]) * 1024
        return fields
    except (FileNotFoundError, ProcessLookupError):
        return {}


def run(pid, output):
    output = Path(output)
    allowed = Path('/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/mini/p2').resolve()
    if not output.resolve().is_relative_to(allowed):
        raise ValueError('observer output must remain inside P2 mini')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if 'tests/boundary_three/p2_maxk.py' not in command or '--base' in command:
        raise ValueError('PID is not the P2 maximum-K driver')
    record = dict(driver_pid=pid, command=command, status='observing', interval_seconds=.05,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  bases={}, peak_sampled_driver_and_worker_RSS_bytes=0,
                  semantics='sampled /proc VmHWM and concurrent RSS; exact compute-stage HWM lives in each numerical receipt; sampling cannot establish an unsampled lifetime maximum')
    last_write = 0
    while Path(f'/proc/{pid}').exists():
        parent = status(pid)
        total = parent.get('VmRSS', 0)
        try:
            children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            break
        for child in children:
            try:
                args = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
                base = args[args.index(b'--base') + 1].decode()
                sample = status(child)
            except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
                continue
            if base not in ('BASE_MATCH', 'V2_MORPH'):
                continue
            item = record['bases'].setdefault(base, dict(pid=int(child), samples=0,
                peak_sampled_VmHWM_bytes=0, peak_sampled_RSS_bytes=0))
            item['samples'] += 1
            item['peak_sampled_VmHWM_bytes'] = max(item['peak_sampled_VmHWM_bytes'], sample.get('VmHWM', 0))
            item['peak_sampled_RSS_bytes'] = max(item['peak_sampled_RSS_bytes'], sample.get('VmRSS', 0))
            item['last_sample_unix'] = time.time()
            total += sample.get('VmRSS', 0)
        record['peak_sampled_driver_and_worker_RSS_bytes'] = max(
            record['peak_sampled_driver_and_worker_RSS_bytes'], total)
        if time.monotonic() - last_write > 5:
            temp = output.with_suffix('.tmp')
            temp.write_text(json.dumps(record, indent=2) + '\n')
            os.replace(temp, output)
            last_write = time.monotonic()
        time.sleep(.05)
    record['status'] = 'finished'
    output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.pid, args.output)
