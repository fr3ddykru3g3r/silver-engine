"""Generate receipt-bound synthetic failure replays and a standalone SVG.

Run as a module from the workspace root. No scientific dataset is read.
"""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
from iris_report.iris_sep.tests.test_pilot_replay import cases
from iris_report.iris_sep.src.iris_sep.pilot_replay import replay_forecast


def build(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (request, expected) in cases().items():
        result = replay_forecast(**copy.deepcopy(request))
        assert result['forecast_status'] == expected
        rows.append({'case': name, 'expected_status': expected, 'forecast': result})
    payload = json.dumps({'scope': 'SYNTHETIC_FIXTURE_ONLY', 'rows': rows}, indent=2, allow_nan=False).encode()
    prediction = output/'fixture_replays.json'; prediction.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    # Plot only a verified serialized artifact, rather than live in-memory values.
    if hashlib.sha256(prediction.read_bytes()).hexdigest() != digest: raise ValueError('replay mutation')
    stored = json.loads(prediction.read_bytes())
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        '<rect width="960" height="540" fill="#f5f3ec"/>',
        '<g font-family="sans-serif" fill="#172b34">',
        '<text x="30" y="36" font-size="23">IRIS-SEP · Synthetic failure replay</text>',
        '<text x="30" y="63" font-size="15">Engineering fixtures only · No observed SEP events or benchmark results</text>']
    for index, row in enumerate(stored['rows']):
        y=100+index*35; forecast=row['forecast']; state=forecast['forecast_status']
        color={'VALID':'#27715a','DEGRADED':'#906500','ABSTAIN':'#9d3838'}[state]
        elements += [f'<text x="30" y="{y}" font-size="15">{html.escape(row["case"])}</text>',
                     f'<text x="265" y="{y}" fill="{color}" font-size="15">{state}</text>',
                     f'<text x="385" y="{y}" font-size="12">{html.escape(", ".join(forecast["abstention_reasons"]) or "fixture inputs accepted")}</text>']
    elements += ['<text x="30" y="515" font-size="12">Every abstention suppresses probability and advisory state. No spacecraft control.</text></g></svg>']
    plot=output/'fixture_replays.svg'; plot.write_text('\n'.join(elements))
    sources=[Path(__file__), Path('iris_report/iris_sep/src/iris_sep/pilot_replay.py'),Path('iris_report/iris_sep/tests/test_pilot_replay.py')]
    receipt={'scope':'SYNTHETIC_FIXTURE_ONLY_NOT_FINAL_RESULTS','locked_test_accessed':False,
        'cases':len(rows),'all_expected_statuses_pass':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (prediction,plot)},
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),indent=2))
