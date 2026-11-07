"""English report retaining every policy, condition and development/transfer split."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from experiments.mission_runtime_profile import read,save,sha

LABELS={'quality_current':'Quality + current intent','quality_initial':'Quality + initial intent',
    'risk_current':'Quality + error budget + current intent','risk_initial':'Quality + error budget + initial intent',
    'fastest_current':'Fastest feasible','fixed_local_yolo':'Fixed local YOLO',
    'fixed_local_mask':'Fixed local Mask R-CNN','fixed_local_cabinet':'Fixed local CABiNet',
    'fixed_remote_mask':'Fixed remote Mask R-CNN'}
ORDER=list(LABELS)


def aggregate(rows):
    if not rows:raise ValueError('Nonempty group required')
    return dict(runs=len(rows),successes=sum(r['dynamic_mission_success'] for r in rows),
        success_rate=sum(r['dynamic_mission_success'] for r in rows)/len(rows),
        deadline_misses=sum(r['task_metrics']['perception']['deadline_misses'] for r in rows),
        mean_nonflight_energy_j=statistics.mean(r['nonflight_energy_j'] for r in rows),
        failed_checks=dict(Counter(k for r in rows for k,v in r['window_metrics'][0]['checks'].items() if not v)),
        battery_failures=sum('battery_constraint_success' in r['failed_constraints'] for r in rows))


def build(root,output):
    root,output=Path(root),Path(output);p=read(root/'plan.json');rows=read(root/'rows.json');audit=read(root/'audit.json')
    assert audit['passed'] and audit['rows_sha256']==sha(root/'rows.json') and audit['plan_sha256']==sha(root/'plan.json')
    assert len(rows)==648
    table=[]
    for split in ('development','transfer','all'):
        for schedule in ('all',*p['intent_schedules']):
            for policy in ORDER:
                sub=[r for r in rows if r['policy']==policy and (split=='all' or r['split']==split) and (schedule=='all' or r['intent_schedule']==schedule)]
                table.append(dict(split=split,schedule=schedule,policy=policy,**aggregate(sub)))
    pairs=[];idx={r['run_id']:r for r in rows}
    for r in rows:
        if r['policy'] not in ('quality_current','risk_current'):continue
        initial=idx[r['run_id'].removesuffix('_current')+'_initial']
        pairs.append(dict(split=r['split'],flight=r['flight'],schedule=r['intent_schedule'],link=r['link'],energy_condition=r['energy_condition'],family=r['policy'].removesuffix('_current'),
            current_success=r['dynamic_mission_success'],initial_success=initial['dynamic_mission_success'],
            mission_delta=int(r['dynamic_mission_success'])-int(initial['dynamic_mission_success']),
            missed_request_delta=r['task_metrics']['perception']['deadline_misses']-initial['task_metrics']['perception']['deadline_misses']))
    matched=[]
    for split in ('development','transfer','all'):
        for schedule in p['intent_schedules']:
            for family in ('quality','risk'):
                sub=[r for r in pairs if r['family']==family and r['schedule']==schedule and (split=='all' or r['split']==split)]
                matched.append(dict(split=split,schedule=schedule,family=family,pairs=len(sub),
                    current_successes=sum(r['current_success'] for r in sub),initial_successes=sum(r['initial_success'] for r in sub),
                    wins=sum(r['mission_delta']>0 for r in sub),losses=sum(r['mission_delta']<0 for r in sub),ties=sum(r['mission_delta']==0 for r in sub),
                    net_missed_requests=sum(r['missed_request_delta'] for r in sub)))
    factors=[]
    for policy in ORDER:
        for link in ('good','degraded'):
            for energy in ('ample','residual_250j'):
                sub=[r for r in rows if r['split']=='transfer' and r['policy']==policy and r['link']==link and r['energy_condition']==energy]
                factors.append(dict(policy=policy,link=link,energy=energy,**aggregate(sub)))
    summary=dict(schema_version='cumulative_dynamic_intent_report_v2',overall=aggregate(rows),aggregates=table,matched=matched,paired_cases=pairs,transfer_factors=factors,audit=audit,
        scope=p['scope'],limits=['All clips and their previous model outcomes were already known. The transfer split is a frozen replay check, not an unseen test set.',
            'This is a new cumulative inspection task. Its success rate must not be presented as improving the old three-resurvey task.',
            'No new Gazebo acquisition or neural inference. Original RGB and real model prediction caches are reused.',
            'Remote service, network and energy are declared cost models. No physical server or battery measurement.',
            'Only six recordings on known terrain; replay cells are correlated. No population significance claim.',
            'Urgency modifies new perception-request deadlines; target identity, quality and total reporting deadline stay fixed.'])
    output.mkdir(parents=True,exist_ok=True);save(output/'summary.json',summary)
    with (output/'results.csv').open('w',newline='') as f:
        keys=['split','schedule','policy','runs','successes','success_rate','deadline_misses','mean_nonflight_energy_j','battery_failures'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(table)
    body='<h1>One mission, changing urgency</h1><p>648 paid perception-routing replays. All policies, failed cases and conditions are retained.</p>'
    body+='<h2>What was changed, and why</h2><p>The earlier task required three fresh status surveys, one every 12 seconds. That is different from continuing one inspection while becoming more urgent. Version 2 preserves evidence, tracking and reports across urgency updates. It requires two of three people to be confirmed from at least two captures separated by two seconds and reported by 44 seconds. The whole flight is charged. A single global cap of two raw false components, two false reports and two duplicate reports is enforced.</p>'
    body+='<p>The previous 0/504 result remains valid for its original periodic-survey contract. Version 2 is not a rescore of those runs: all policies execute again under the public cumulative intent, with continuous downstream state.</p>'
    body+='<h2>Frozen design</h2><p>6 recordings × 3 urgency schedules × 9 policies × 2 links × 2 battery conditions = 648 runs. Schedules: 8 s throughout; 8→2 s at 12 s; 8→2→8 s at 12/24 s. Two shoreline recordings form the development split; four wide-area/revisit recordings form the transfer split. These are all previously evaluated recordings. Policies and conditions were locked before the new development and transfer outcomes.</p>'
    body+='<p>The added error-budget router filters candidates whose expected false components over the mission exceed the public cap, then chooses the highest estimated recall among candidates satisfying time, link and battery guards. Its initial-intent counterpart shares the same rule. This is a prior-based heuristic, not a guaranteed error bound.</p>'
    body+='<h2>Complete-mission success</h2><img src="mission_success.png" alt="Mission success by split and policy">'
    for split in ('transfer','development','all'):
        body+=f'<h3>{split.title()} — all schedules</h3><table><tr><th>Router</th><th>Successful missions</th><th>Success rate</th><th>Missed requests</th><th>Mean nonflight energy (J)</th></tr>'
        for r in table:
            if r['split']==split and r['schedule']=='all':body+=f'<tr><td>{LABELS[r["policy"]]}</td><td>{r["successes"]}/{r["runs"]}</td><td>{r["success_rate"]:.1%}</td><td>{r["deadline_misses"]}</td><td>{r["mean_nonflight_energy_j"]:.1f}</td></tr>'
        body+='</table>'
    body+='<h2>Does receiving intent updates improve full missions?</h2><p>Same router, recording, link, battery and actual task requirements. Only update visibility differs. A win means current intent succeeds and initial intent fails.</p><table><tr><th>Split</th><th>Schedule</th><th>Router</th><th>Current / initial successes</th><th>Wins / losses / ties</th></tr>'
    for r in matched:
        if r['split']=='all':continue
        body+=f'<tr><td>{r["split"]}</td><td>{r["schedule"]}</td><td>{r["family"]}</td><td>{r["current_successes"]} / {r["initial_successes"]} of {r["pairs"]}</td><td>{r["wins"]} / {r["losses"]} / {r["ties"]}</td></tr>'
    body+='</table><img src="matched_intent.png" alt="Matched intent visibility comparisons">'
    body+='<h2>What the results establish</h2><p>On the four transfer recordings, error-budget routing with current intent succeeds in 25/48 missions, compared with 22/48 for fixed remote Mask R-CNN and 6/48 for fixed local Mask R-CNN. The strongest fixed baseline is only three missions (6.25 percentage points) behind. This is a modest observed difference, not evidence of statistical significance.</p><p>Updating intent alone has limited benefit here: the quality pair both succeed in 24/48, and the error-budget pair changes from 24/48 to 25/48. Both current-intent routers reduce missed requests from 90 to zero. In the development split, quality routing with current intent loses one mission relative to initial intent. Faster deadline compliance does not automatically mean higher mission success.</p><p>The single transfer win for intent-aware error-budget routing occurs in sector_revisit/sector_7, urgency recovery, degraded link and ample battery. It uses local Mask R-CNN outside the urgent interval and local YOLO during it, confirming/reporting two focal people. Its initial-intent counterpart keeps local Mask R-CNN, misses three urgent requests and reports only one focal person. The quality-only current router chooses CABiNet during urgency and reports one focal person. This is a post-hoc illustration; the full table retains every case.</p>'
    body+='<h2>Network and battery conditions</h2><table><tr><th>Router</th><th>Link</th><th>Battery</th><th>Transfer success</th></tr>'
    for r in factors:body+=f'<tr><td>{LABELS[r["policy"]]}</td><td>{r["link"]}</td><td>{r["energy"]}</td><td>{r["successes"]}/{r["runs"]}</td></tr>'
    body+='</table><h2>Limits</h2><ul>'+''.join('<li>'+s+'</li>' for s in summary['limits'])+'</ul>'
    body+=f'<p>Receipt audit: {audit["decisions"]:,} decisions, {audit["matched_prefix_checks"]} matched histories. This reconstructs timing, costs and mission scoring; it does not independently rerun neural inference.</p><p><a href="summary.json">All results and matched cases</a> · <a href="results.csv">Result table (CSV)</a></p>'
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Cumulative mission routing</title><style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#18334c}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce4eb}th{background:#eef5fb}img{width:100%}</style><body>'+body+'</body></html>')
    print(json.dumps([r for r in table if r['split']=='transfer' and r['schedule']=='all'],indent=2))

def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);s=read(output/'summary.json')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for ax,split in zip(axes,('development','transfer')):
        rows=[next(r for r in s['aggregates'] if r['split']==split and r['schedule']=='all' and r['policy']==p) for p in ORDER]
        colors=['#2469a0','#b7c7d5','#238b70','#bad9cf']+['#8393a3']*5
        ax.barh(range(len(rows)),[100*r['success_rate'] for r in rows],color=colors)
        ax.set_yticks(range(len(rows)));ax.set_yticklabels([LABELS[p] for p in ORDER]);ax.set_ylim(len(rows)-.5,-.5);ax.set_xlim(0,125);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Complete-mission success (%)')
        for i,r in enumerate(rows):ax.text(100*r['success_rate']+1.5,i,f'{r["successes"]}/{r["runs"]}',va='center')
        ax.set_title(split.title()+(' (2 known recordings)' if split=='development' else ' (4 known recordings)'));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Cumulative inspection under changing per-request urgency',fontsize=17)
    fig.text(.02,.01,'Same flight and predictions for every router. All schedules, links and batteries retained. Transfer uses previously evaluated clips; no unseen-data claim.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    for ext in ('png','svg'):fig.savefig(output/f'mission_success.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    schedules=['no_change','urgency_increase','urgency_recovery']
    for ax,family in zip(axes,('quality','risk')):
        rows=[next(r for r in s['matched'] if r['split']=='transfer' and r['family']==family and r['schedule']==k) for k in schedules]
        for off,key,color,label in [(-.18,'current_successes','#2469a0','Current intent'),(.18,'initial_successes','#d88b39','Initial intent only')]:
            bars=ax.bar([i+off for i in range(3)],[100*r[key]/r['pairs'] for r in rows],.34,color=color,label=label)
            for b,r in zip(bars,rows):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{r[key]}/{r["pairs"]}',ha='center',fontsize=10)
        ax.set_xticks(range(3));ax.set_xticklabels(['No change','Urgency\nincrease','Urgency\nrecovery']);ax.set_ylim(0,115);ax.set_title('Quality router' if family=='quality' else 'Quality + error-budget router')
    axes[0].set_ylabel('Complete-mission success (%)')
    fig.suptitle('Matched update visibility on four transfer recordings',fontsize=15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1))
    for ext in ('png','svg'):fig.savefig(output/f'matched_intent.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/intent_mission_v2_20261031'));p.add_argument('--output',type=Path,default=Path('reports/intent_mission_v2_20261031'));p.add_argument('--figures-only',action='store_true');a=p.parse_args()
    if a.figures_only:figures(a.output)
    else:build(a.root,a.output)
