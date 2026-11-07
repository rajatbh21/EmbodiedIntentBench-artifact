"""Window-local fresh reports plus complete paid mission accounting."""
from experiments.lake_missions.objectives import score as phase_score
from experiments.mission_task_deadlines.study import task_metrics


def combined_reports(reporters, time_s):
    reports=[dict(r,window_id=i) for i,ev in enumerate(reporters) for r in ev.summary(time_s)['reports']]
    return dict(reports=reports,window_count=len(reporters),
        semantics='Report IDs and duplicate suppression are local to each explicitly requested status window.')


def dynamic_outcome(contract, events, result):
    doc=contract.document();focal=set(doc['evaluator_target_ids'])
    phases=phase_score(doc['mission'],events,result)['phases']
    reports=result['report_quality']['reports'];rows=[]
    for i,(p,definition) in enumerate(zip(phases,doc['mission']['phases'])):
        current=[r for r in reports if r['window_id']==i]
        correct=[r['target_id'] for r in current if r['target_id'] is not None]
        # Report support must be fresh even if earlier computation straddles an update.
        valid=[r for r in current if r['target_id'] in focal
            and r['completion_time_s']<=definition['deadline_s']
            and all(definition['start_s']<=e['capture_s']<definition['end_s'] for e in r['evidence'])]
        ids={r['target_id'] for r in valid}
        false=len(current)-len(correct);duplicates=len(correct)-len(set(correct))
        selected=[e for e in events if e['window_id']==i]
        first=min((r['completion_time_s'] for r in valid),default=None)
        checks=dict(repeated_evidence=p['success'],fresh_report_recall=len(ids)/len(focal)>=2/3,
            false_report_limit=false<=2,duplicate_report_limit=duplicates<=2)
        rows.append(dict(window_id=i,start_s=definition['start_s'],end_s=definition['end_s'],
            deadline_s=definition['deadline_s'],checks=checks,success=all(checks.values()),
            focal_correct_reports=len(ids),evidence_recall=p['evidence_recall'],
            world_false_reports=false,world_duplicate_reports=duplicates,
            timely_targets=sorted(ids),first_valid_report_s=first,
            first_valid_report_wait_s=None if first is None else first-definition['start_s'],
            first_decision_wait_s=None if not selected else selected[0]['capture_time_s']-definition['start_s'],
            requests=len(selected),abstentions=sum(e['selected_action_id'] is None for e in selected),
            deadline_misses=sum(e['perception']['deadline_missed'] for e in selected)))
    allowed={'completed','task_deadline_miss','transport_timeout','mission_deadline_exceeded','abstained','timeout','deadline'}
    execution=(result['mission_completed'] and result['termination_reason']=='mission_window_complete'
        and result['error'] is None and result['final_time_s']>=doc['duration_s']
        and all(e['perception']['status'] in allowed and
            (e['perception']['admitted'] or (e['pipeline'] is None and not e['emitted_reports'])) for e in events))
    checks=dict(execution=execution,all_windows=all(w['success'] for w in rows),**result['constraints'])
    success=all(checks.values())
    return dict(schema_version='dynamic_intent_paid_outcome_v1',dynamic_mission_success=success,
        local_inspection_success=success,full_wide_mission_evaluated=False,
        checks=checks,failed_constraints=[k for k,v in checks.items() if not v],
        windows=rows,window_successes=sum(w['success'] for w in rows),phases=phases,
        reporting=dict(world_false_reports=sum(w['world_false_reports'] for w in rows),
            world_duplicates=sum(w['world_duplicate_reports'] for w in rows),
            correct_nonfocal_reports=sum(r['target_id'] is not None and r['target_id'] not in focal for r in reports),
            timely_focal_reports=[(w['window_id'],t) for w in rows for t in w['timely_targets']],
            recall_by_deadline=sum(w['focal_correct_reports'] for w in rows)/(len(focal)*len(rows))),
        task_metrics=task_metrics(events),abstentions=sum(e['selected_action_id'] is None for e in events),
        semantics='All three fixed status windows and resource limits must pass. Missed requests cannot contribute evidence; later valid requests can recover. Prior-window work remains charged and can complete against its original deadline, but cannot satisfy a new window.')
