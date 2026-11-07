"""Describe the locked pilot without treating repeated cells as independent flights."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from experiments.mission_runtime_profile import read,save,sha

LABELS={
 'fixed_local_fast':'Fixed local YOLO11s',
 'fixed_local_heavy':'Fixed local Mask R-CNN union',
 'fixed_remote_union':'Fixed remote Mask R-CNN union',
 'deadline_aware':'Intent-aware heuristic',
 'resource_threshold':'Resource thresholds',
 'accuracy_latency':'Accuracy / latency',
 'supervised_utility':'Supervised utility',
 'linucb_frozen':'Frozen LinUCB',
}


def summarize_rows(rows):
    return dict(runs=len(rows),mission_successes=sum(r['mission_success'] for r in rows),
        completed_sectors=sum(r['successful_phases'] for r in rows),total_sectors=8*len(rows),
        mean_report_recall=statistics.mean(r['report_recall'] for r in rows),
        mean_final_battery_frac=statistics.mean(r['final_battery_frac'] for r in rows),
        min_final_battery_frac=min(r['final_battery_frac'] for r in rows),
        max_final_battery_frac=max(r['final_battery_frac'] for r in rows),
        mean_energy_wh=statistics.mean(r['total_energy_j']/3600 for r in rows),
        mean_nonflight_energy_wh=statistics.mean(r['nonflight_energy_j']/3600 for r in rows),
        mean_communication_mb=statistics.mean(r['communication_mb'] for r in rows),
        decisions=sum(r['decision_count'] for r in rows),
        eligible_opportunities=sum(r['eligible_opportunity_count'] for r in rows),
        busy_skipped_opportunities=sum(r['skipped_opportunity_count'] for r in rows),
        abstentions=sum(r['abstentions'] for r in rows),
        failed_constraints=dict(Counter(k for r in rows for k in r['failed_constraints'])))


def build(root):
    root=Path(root);plan=read(root/'plan.json');completion=read(root/'completion.json')
    if not (completion['complete'] and completion['plan_sha256']==sha(root/'plan.json') and
            completion['run_index_sha256']==sha(root/'run_index.json')):
        raise ValueError('Completed bound replay grid required')
    rows=read(root/'run_index.json')
    if len(rows)!=plan['expected_runs'] or {r['run_id'] for r in rows}!={r['run_id'] for r in plan['grid']}:
        raise ValueError('Incomplete or duplicate replay grid')
    pooled={};per_route={};per_condition={};bindings={}
    for p in plan['factors']['policies']:
        selected=[r for r in rows if r['policy']==p]
        pooled[p]=summarize_rows(selected)
        per_route[p]={f:summarize_rows([r for r in selected if r['flight']==f])
                      for f in plan['factors']['flights']}
        per_condition[p]={}
        for field in ('link','perception_budget_s','battery'):
            levels=sorted({r[field] for r in selected})
            per_condition[p][field]={str(level):summarize_rows([r for r in selected if r[field]==level])
                                      for level in levels}
        actions=Counter();admitted=attempted=missed=0
        for r in selected:
            for filename,digest in r['artifacts_sha256'].items():
                if sha(root/r['path']/filename)!=digest:raise ValueError('Changed run artifact')
            path=root/r['path']/'events.json';events=read(path);bindings[str(path)]=sha(path)
            for e in events:
                if e['selected_action_id'] is not None:
                    actions[e['selected_action_id']]+=1;attempted+=1
                admitted+=e['perception']['admitted']
                missed+=e['perception']['deadline_missed']
        pooled[p].update(selected_actions=dict(actions),attempted_perception=attempted,
                         admitted_perception=admitted,perception_deadline_misses=missed)
    paired={}
    for field in ('link','perception_budget_s','battery'):
        other=[k for k in ('flight','policy','link','perception_budget_s','battery') if k!=field]
        groups={}
        for r in rows:groups.setdefault(tuple(r[k] for k in other),{})[r[field]]=r
        pairs=[]
        for key,g in groups.items():
            a,b=sorted(g)
            pairs.append(dict(context=dict(zip(other,key)),first=a,second=b,
                sector_delta=g[b]['successful_phases']-g[a]['successful_phases'],
                report_recall_delta=g[b]['report_recall']-g[a]['report_recall'],
                perception_deadline_miss_delta=(g[b]['task_metrics']['perception']['deadline_misses']-
                                                g[a]['task_metrics']['perception']['deadline_misses']),
                mission_success_changed=g[a]['mission_success']!=g[b]['mission_success']))
        paired[field]=dict(pairs=len(pairs),sector_changes=sum(x['sector_delta']!=0 for x in pairs),
            report_recall_changes=sum(x['report_recall_delta']!=0 for x in pairs),
            perception_deadline_miss_changes=sum(x['perception_deadline_miss_delta']!=0 for x in pairs),
            mission_success_changes=sum(x['mission_success_changed'] for x in pairs),rows=pairs)
    summary=dict(schema_version='wide_area_pilot_results_v1',plan_sha256=sha(root/'plan.json'),
        run_index_sha256=sha(root/'run_index.json'),source_events_sha256=bindings,
        all=summarize_rows(rows),by_policy=pooled,by_route_and_policy=per_route,
        by_condition_and_policy=per_condition,matched_factor_changes=paired,
        independent_physical_recordings=3,independent_base_terrains=1,
        analysis='Descriptive paired deterministic sensitivity. No confidence intervals or population-generalization claim.',
        energy='Configured model, radio and flight energy; not electrical measurements.',
        source_sha256=sha(Path(__file__)))
    diagnostics={}
    for flight in plan['factors']['flights']:
        path=root/'model_diagnostics'/f'{flight}.json'
        if path.exists():
            d=read(path);bound=d['optimistic_cross_model_evidence_bound']
            diagnostics[flight]=dict(source_sha256=sha(path),
                optimistic_evidence_sufficient_sectors=sum(p['evidence_requirement_met'] for p in bound['phases']),
                failed_sector_ids=bound['failed_phase_ids'],
                pool_evidence_infeasible=not bound['all_phase_evidence_requirements_met'],
                supported_counts=[p['supported_target_count'] for p in bound['phases']])
    summary['model_evidence_diagnostics']=diagnostics
    audit_path=root/'independent_audit.json'
    if audit_path.exists():
        audit=read(audit_path)
        if not (audit['passed'] and audit['verified_runs']==len(rows) and
                audit['plan_sha256']==summary['plan_sha256'] and
                audit['run_index_sha256']==summary['run_index_sha256']):
            raise ValueError('Independent audit does not certify this completed grid')
        summary['execution_audit']=dict(passed=True,source_sha256=sha(audit_path),
            verified_runs=audit['verified_runs'],verified_events=audit['verified_events'],
            verified_selected_cache_reads=audit['verified_selected_cache_reads'])
    else:
        summary['execution_audit']=dict(passed=False,status='Independent execution audit pending')
    save(root/'summary.json',summary)
    lines=['WIDE-AREA PERCEPTION ROUTING: FIRST SCHEDULED-INSPECTION PILOT','',
        '192 paired replays = 3 fixed flights x 8 policies x 2 links x 2 task budgets x 2 initial charges.',
        'Each flight has 8 authored inspection windows and 64 eligible original captures. Transit remains in the flight clock and battery accounting.',
        '9 perception configurations / 18 local-or-virtual-remote actions. Native RGB inference was actually executed for every eligible frame and configuration.',
        (f"Independent execution audit: PASSED, {summary['execution_audit']['verified_runs']} runs / {summary['execution_audit']['verified_events']} events."
         if summary['execution_audit']['passed'] else 'Independent execution audit: PENDING.'),
        '', 'MISSION CRITERIA', plan['mission_criteria'],'',
        'POOLED RESULTS (24 runs and 192 sectors per policy)',
        'Policy | Full missions | Sectors | Mean report recall | Admitted perception | Deadline misses | Mean nonflight Wh']
    for p,s in pooled.items():
        lines.append(f"{LABELS[p]} | {s['mission_successes']}/{s['runs']} | {s['completed_sectors']}/{s['total_sectors']} | {s['mean_report_recall']:.1%} | {s['admitted_perception']}/{s['attempted_perception']} | {s['perception_deadline_misses']} | {s['mean_nonflight_energy_wh']:.3f}")
    for flight in plan['factors']['flights']:
        lines+=['',flight.upper(),'Policy | Full missions / 8 | Sectors / 64 | Mean report recall']
        for p in pooled:
            s=per_route[p][flight]
            lines.append(f"{LABELS[p]} | {s['mission_successes']}/8 | {s['completed_sectors']}/64 | {s['mean_report_recall']:.1%}")
    lines+=['','MATCHED SENSITIVITY']
    for field,s in paired.items():
        lines.append(f"{field}: {s['mission_success_changes']}/{s['pairs']} paired full-success changes; {s['sector_changes']}/{s['pairs']} sector-count changes; {s['report_recall_changes']}/{s['pairs']} report-recall changes; {s['perception_deadline_miss_changes']}/{s['pairs']} deadline-miss-count changes.")
    lines+=['',f"Observed final battery range: {summary['all']['min_final_battery_frac']:.1%} to {summary['all']['max_final_battery_frac']:.1%}; configured reserve20%."]
    lines+=['','INTERPRETATION AND LIMITS',
        'Full mission success requires every sector to pass. Sector completion and report recall reveal partial progress but do not replace that endpoint.',
        'The flight path and duration are fixed across routers. Task latency, dropped outputs and missed observation opportunities explain timing effects.',
        'The intermittent link is an authored virtual trace. No hardware network or drone power measurements are claimed.',
        plan['battery'],plan['forecast'],plan['interpretation'],plan['design_chronology'],
        'The learned policies were frozen before this study; no evaluation labels or reward feedback update them.',
        'Model diagnostics are reported separately without changing the frozen mission or discarding difficult sectors.']
    if diagnostics:
        lines+=['','MODEL-POOL NECESSARY EVIDENCE BOUND',
            'This optimistic evaluator-only bound combines possible matched target IDs across all nine models at every opportunity. It ignores paid timing and false positives; it is not an executable router. Failure proves this fixed pool cannot complete the locked intent on these opportunities. Passing would not prove mission feasibility.']
        for flight,d in diagnostics.items():
            lines.append(f"{flight}: {d['optimistic_evidence_sufficient_sectors']}/8 sectors have sufficient optimistic repeated evidence; supported people by sector={d['supported_counts']}; pool evidence infeasible={d['pool_evidence_infeasible']}.")
        if any(d['pool_evidence_infeasible'] for d in diagnostics.values()):
            lines.append('These cases cannot demonstrate differences in binary full-mission success. Partial-sector, report, deadline and resource differences remain descriptive; zero successes do not establish router equivalence. Future feasibility repair requires a separate development protocol and fresh evaluation cohort, not relaxed thresholds on these results.')
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:20]))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    build(parser.parse_args().root)
