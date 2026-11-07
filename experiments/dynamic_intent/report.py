"""English report with complete denominators and matched intent-update contrasts."""
from collections import defaultdict, Counter
from pathlib import Path
import argparse
import html
import json
import statistics
from experiments.mission_runtime_profile import read, save, sha

LABELS={'quality_current':'Quality + current intent','quality_initial':'Quality + initial intent',
 'fastest_current':'Fastest feasible','fixed_local_yolo':'Fixed local YOLO',
 'fixed_local_mask':'Fixed local Mask R-CNN','fixed_local_cabinet':'Fixed local CABiNet',
 'fixed_remote_mask':'Fixed remote Mask R-CNN'}
ORDER=list(LABELS)


def aggregate(rows):
    windows=[w for r in rows for w in r['window_metrics']]
    waits=[w['first_valid_report_wait_s'] for w in windows if w['first_valid_report_wait_s'] is not None]
    requests=sum(w['requests']-w['abstentions'] for w in windows)
    misses=sum(w['deadline_misses'] for w in windows)
    return dict(runs=len(rows),mission_successes=sum(r['dynamic_mission_success'] for r in rows),
        windows=len(windows),window_successes=sum(r['window_successes'] for r in rows),
        dispatched_requests=requests,deadline_misses=misses,
        request_miss_rate=misses/requests if requests else None,
        mean_energy_j=statistics.mean(r['total_energy_j'] for r in rows),
        mean_nonflight_energy_j=statistics.mean(r['nonflight_energy_j'] for r in rows),
        report_wait_median_s=statistics.median(waits) if waits else None,
        windows_with_valid_reports=len(waits),windows_without_valid_reports=len(windows)-len(waits))


def build(root, output):
    root,output=Path(root),Path(output);plan=read(root/'plan.json');rows=read(root/'rows.json');audit=read(root/'audit.json')
    assert audit['passed'] and audit['plan_sha256']==sha(root/'plan.json') and audit['rows_sha256']==sha(root/'rows.json')
    assert len(rows)==504
    gate=read(root/'sensor_gate.json');possible={r['flight'] for r in gate['cases'] if r['sensor_possible']}
    table=[]
    diagnosis=[]
    for policy in ORDER:
        subset=[r for r in rows if r['policy']==policy]
        for window_id in range(3):
            windows=[r['window_metrics'][window_id] for r in subset]
            diagnosis.append(dict(policy=policy,window_id=window_id,runs=len(windows),
                successes=sum(w['success'] for w in windows),
                failed_checks=dict(Counter(k for w in windows for k,v in w['checks'].items() if not v))))
    for scope in ('all_clips','sensor_possible_clips'):
      for schedule in plan['intent_schedules']:
       for name in ORDER:
        subset=[r for r in rows if r['intent_schedule']==schedule and r['policy']==name and (scope=='all_clips' or r['flight'] in possible)]
        row=dict(scope=scope,schedule=schedule,policy=name,**aggregate(subset));table.append(row)
    indexed={(r['flight'],r['intent_schedule'],r['link'],r['energy_condition'],r['policy']):r for r in rows}
    contrasts=[]
    for key,current in indexed.items():
        if key[-1]!='quality_current':continue
        initial=indexed[(*key[:-1],'quality_initial')]
        contrasts.append(dict(flight=key[0],schedule=key[1],link=key[2],energy_condition=key[3],
            sensor_possible=key[0] in possible,
            mission_delta=int(current['dynamic_mission_success'])-int(initial['dynamic_mission_success']),
            window_delta=current['window_successes']-initial['window_successes'],
            deadline_miss_delta=sum(w['deadline_misses'] for w in current['window_metrics'])-sum(w['deadline_misses'] for w in initial['window_metrics']),
            onboard_nonflight_energy_delta_j=current['nonflight_energy_j']-initial['nonflight_energy_j'],
            action_counts_changed=current['selected_action_counts']!=initial['selected_action_counts']))
    groups=[]
    for schedule in plan['intent_schedules']:
        group=[c for c in contrasts if c['schedule']==schedule]
        groups.append(dict(schedule=schedule,pairs=len(group),wins=sum(c['mission_delta']>0 for c in group),
            losses=sum(c['mission_delta']<0 for c in group),ties=sum(c['mission_delta']==0 for c in group),
            net_completed_windows=sum(c['window_delta'] for c in group),
            net_request_misses=sum(c['deadline_miss_delta'] for c in group),
            changed_action_counts=sum(c['action_counts_changed'] for c in group)))
    output.mkdir(parents=True,exist_ok=True)
    summary=dict(scope=plan['scope'],runs=504,recordings=6,sensor_possible_recordings=4,
        aggregates=table,paired_contrasts=contrasts,matched_summary=groups,audit=audit,
        overall=aggregate(rows),window_failure_diagnosis=diagnosis,
        claim_limits=['Previously evaluated recordings and frozen model predictions; no new physical recording or neural inference.',
            'Four clips pass an optimistic visibility bound, not a paid-policy feasibility guarantee. Two impossible clips remain in all-clips results.',
            'Quality and repeated-evidence requirements are constant across schedules. Three identical report windows also apply to no_change.',
            'Battery, network and server costs are modeled. Correlated replay cells are not independent field trials.',
            'Latency statistics exclude missing reports only when explicitly labeled and disclose their count.'])
    save(output/'summary.json',summary)
    body='<h1>Changing intent during flight</h1><p class="lead">A perception-routing development pilot: 504 paid replays on six previously recorded flights.</p>'
    body+='<p><strong>Result: 0/504 complete missions; 172/1,512 successful reporting windows.</strong> Receiving updated intent reduced deadline misses and added one successful window in each changing schedule, compared with retaining initial intent. This pilot does not demonstrate improved complete-mission success.</p>'
    body+='<h2>What changes</h2><table><tr><th>Schedule</th><th>0-12 seconds</th><th>12-24 seconds</th><th>24-36 seconds</th></tr><tr><td>No change</td><td>8 s per request</td><td>8 s</td><td>8 s</td></tr><tr><td>Urgency increase</td><td>8 s</td><td>2 s</td><td>2 s</td></tr><tr><td>Urgency recovery</td><td>8 s</td><td>2 s</td><td>8 s</td></tr></table>'
    body+='<p>Each observation window requests fresh reports for at least two of three sector targets, each supported by two captures at least two seconds apart. Quality requirements stay unchanged. Reporting has an equal eight-second delivery allowance. Flight and battery accounting continue through the full 42-second recording. A request already running keeps its original deadline.</p>'
    body+='<h2>Matched comparison: current vs initial intent</h2><p>Both routers use the same frozen quality priors and resource guards. Only the perceived request budget changes. No-change runs and the pre-change traces agree exactly.</p><table><tr><th>Schedule</th><th>Mission wins / losses / ties</th><th>Net completed windows</th><th>Net missed requests</th></tr>'
    for r in groups:body+=f'<tr><td>{r["schedule"]}</td><td>{r["wins"]} / {r["losses"]} / {r["ties"]} ({r["pairs"]} pairs)</td><td>{r["net_completed_windows"]:+d}</td><td>{r["net_request_misses"]:+d}</td></tr>'
    body+='</table><p>A model switch is not itself a success. Negative results and cases with no useful change are retained.</p><img src="mission_success.svg" alt="Complete mission success"><img src="window_success.svg" alt="Completed windows">'
    for scope in ('all_clips','sensor_possible_clips'):
        body+='<h2>'+scope.replace('_',' ').title()+'</h2><table><tr><th>Schedule</th><th>Policy</th><th>Mission success</th><th>Windows</th><th>Missed / dispatched requests</th><th>Mean nonflight energy (J)</th><th>Median report wait (s)*</th><th>Windows without valid report</th></tr>'
        for r in table:
            if r['scope']!=scope:continue
            wait='N/A' if r['report_wait_median_s'] is None else f'{r["report_wait_median_s"]:.2f}'
            body+=f'<tr><td>{r["schedule"]}</td><td>{LABELS[r["policy"]]}</td><td>{r["mission_successes"]}/{r["runs"]}</td><td>{r["window_successes"]}/{r["windows"]}</td><td>{r["deadline_misses"]}/{r["dispatched_requests"]}</td><td>{r["mean_nonflight_energy_j"]:.1f}</td><td>{wait}</td><td>{r["windows_without_valid_reports"]}/{r["windows"]}</td></tr>'
        body+='</table><p>* Among windows with a valid focal report; failures are counted in the adjacent column.</p>'
    body+='<h2>Example of a change during flight</h2><p>Preselected example: sector_revisit/sector_3, degraded link, ample battery, urgency recovery. Bars show paid perception attempts; red outlines mark missed deadlines. It is an illustrative case, not a selected best-performing result.</p><img src="intent_timeline.svg" alt="Intent update and model-choice timeline">'
    body+='<h2>Why complete missions still fail</h2><p>Repeated evidence and fresh correct reports are the failed checks for the matched quality routers, especially in the first and last windows. These counts describe overlapping failures, not independent causes. Each row pools 72 correlated runs across schedules, links and battery conditions.</p><table><tr><th>Policy</th><th>Window</th><th>Passed</th><th>Repeated-evidence failures</th><th>Fresh-report failures</th></tr>'
    for r in diagnosis:
        if r['policy'] not in ('quality_current','quality_initial'):continue
        body+=f'<tr><td>{LABELS[r["policy"]]}</td><td>{r["window_id"]+1}</td><td>{r["successes"]}/{r["runs"]}</td><td>{r["failed_checks"].get("repeated_evidence",0)}</td><td>{r["failed_checks"].get("fresh_report_recall",0)}</td></tr>'
    body+='</table><p>Next: acquire longer repeated views and check that a cost-paying policy can satisfy the reporting contract on development recordings before freezing a new evaluation set. Keep these negative results; do not retroactively relax thresholds to turn them into successes.</p>'
    body+='<h2>Limits and audit</h2><ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in summary['claim_limits'])+'</ul><p>'+html.escape(audit['scope'])+'</p><p><a href="summary.json">Full aggregates and all matched pairs (JSON)</a></p>'
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dynamic intent pilot</title><style>body{max-width:1450px;margin:32px auto;padding:0 24px;font:16px/1.5 system-ui;color:#18334c}h1{font-size:34px}img{width:100%;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce4eb}th{background:#eef5fb}.lead{font-size:20px}</style><body>'+body+'</body></html>')
    print(json.dumps(groups,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/dynamic_intent_v1_20261031'));p.add_argument('--output',type=Path,default=Path('reports/dynamic_intent_v1_20261031'));a=p.parse_args();build(a.root,a.output)
