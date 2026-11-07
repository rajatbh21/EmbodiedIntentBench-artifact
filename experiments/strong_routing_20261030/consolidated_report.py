"""Read-only consolidation of both completed prospective routing rounds.

Never consumes partial round-two outcomes. All strict results remain primary;
registered endpoint sensitivity is separately rescored from original executions.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
from itertools import combinations
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from experiments.strong_routing_20261030 import report as base

REPO=base.REPO
ROUND2=REPO/"results/strong_routing_round2_comparison_v1_20261030"
NATIVE2=REPO/"results/strong_routing_round2_native_v1_20261030"
ENDPOINT1=REPO/"results/strong_routing_endpoint_sensitivity_v1_20261030"
OUTPUT=REPO/"reports/strong_routing_consolidated_20261030"
LABELS={**base.LABELS,
    "fixed_remote_yolo":"Fixed remote YOLO",
    "calibrated_quality":"Calibrated quality (162 frames)",
    "calibrated_evidence_rate":"Evidence rate (162 frames)",
    "previous_calibrated_evidence_rate":"Previous evidence rate (54 frames)",
    "mission_risk_budget":"Mission risk budget (162 frames)"}
RULE="Separately rescore original execution by removing only raw-component cap from sector evidence; keep repeated evidence, timely correct reporting, false/duplicate report caps and all resource checks."


def bind(path,hashes,expected=None):
    path=Path(path).resolve()
    value=hashes.get(str(path)) or base.sha(path)
    base.require(expected is None or value==expected,"Changed report source: "+str(path))
    hashes[str(path)]=value
    return value


def registered_rescore(hashes):
    """Execute the exact registered function without importing simulator packages."""
    plan_path=ENDPOINT1/"plan.json"
    plan=base.read(plan_path);bind(plan_path,hashes)
    for path,expected in plan["implementation_sha256"].items():bind(path,hashes,expected)
    for path,expected in plan["motivating_control_sha256"].items():bind(path,hashes,expected)
    source=REPO/"experiments/strong_routing_20261030/endpoint_sensitivity.py"
    base.require(str(source) in plan["implementation_sha256"],"Registered endpoint implementation missing")
    parsed=ast.parse(source.read_text())
    functions=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name=="rescore"]
    base.require(len(functions)==1,"Unique registered rescore function required")
    namespace={"require":base.require}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),"exec"),namespace)
    return namespace["rescore"],plan


def endpoints(source,rows,rescore,hashes):
    values=[]
    for row in rows:
        path=Path(source)/row["path"]/"outcome.json"
        bind(path,hashes,row["artifacts_sha256"]["outcome.json"])
        result=rescore(base.read(path))
        base.require(result["original_success"]==row["local_inspection_success"],"Endpoint index disagreement")
        values.append({key:row[key] for key in ("run_id","flight","policy","link","energy_condition","perception_budget_s")}
                      |result)
    return values


def equivalence_and_ablation(rows,traces,policies):
    lookup=defaultdict(dict)
    for row in rows:
        key=(row["flight"],row["link"],row["perception_budget_s"],row["energy_condition"])
        lookup[key][row["policy"]]=row
    base.require(len(lookup)==72 and all(set(pair)==set(policies) for pair in lookup.values()),"Incomplete paired trajectories")
    pairs=[];groups={p:{p} for p in policies}
    for a,b in combinations(policies,2):
        counts=Counter()
        for pair in lookup.values():
            ra,rb=pair[a],pair[b]
            sa,sb=ra["local_inspection_success"],rb["local_inspection_success"]
            counts.update({"both_success":int(sa and sb),"both_failure":int(not sa and not sb),
                "a_only_success":int(sa and not sb),"b_only_success":int(sb and not sa),
                "identical_capture_action_admission_traces":int(traces[ra["run_id"]]==traces[rb["run_id"]]),
                "identical_nonflight_energy":int(base.close(ra["nonflight_energy_j"],rb["nonflight_energy_j"]))})
        row=dict(policy_a=a,policy_b=b,paired_conditions=72,**counts)
        pairs.append(row)
        if counts["identical_capture_action_admission_traces"]==72:
            union=groups[a]|groups[b]
            for p in union:groups[p]=union
    unique={tuple(p for p in policies if p in group) for group in groups.values() if len(group)>1}
    return dict(pairs=pairs,identical_trajectory_groups=[list(g) for g in sorted(unique)],
        scope="Complete capture/action/placement/admission/status sequences, including abstention. All 72 paired conditions; no independent sample claim.")


def risk_diagnostics(source,rows,hashes):
    counts=defaultdict(Counter)
    for row in rows:
        if row["policy"]!="mission_risk_budget":continue
        path=Path(source)/row["path"]/"events.json";bind(path,hashes,row["artifacts_sha256"]["events.json"])
        for event in base.read(path):
            decision=event["router_decision"]
            base.require(decision["mission_risk_guard"]["uses_actual_feedback"] is False,
                         "Mission risk heuristic must not consume correctness feedback")
            for candidate in decision["candidates"]:
                counter=counts[candidate["action_id"]]
                counter["candidate_assessments"]+=1
                counter["resource_feasible_assessments"]+=candidate["feasible"]
                counter["expected_risk_exclusions_after_resource_guard"]+=bool(candidate["feasible"] and not candidate["mission_risk_feasible"])
                counter["selected_decisions"]+=event["selected_action_id"]==candidate["action_id"]
    return {key:dict(value) for key,value in counts.items()}


def error_outcomes(source,rows,hashes):
    counts=defaultdict(Counter)
    for row in rows:
        path=Path(source)/row["path"]/"intent.json"
        bind(path,hashes,row["artifacts_sha256"]["intent.json"])
        intent=base.read(path)
        base.require(len(intent["mission"]["phases"])==1,"Error counts require one local inspection phase")
        counter=counts[row["policy"]]
        counter["runs"]+=1
        counter["raw_component_cap_violating_runs"]+=row["raw_false_components"]>intent["mission"]["phases"][0]["max_false_components"]
        counter["false_report_cap_violating_runs"]+=row["world_false_reports"]>intent["reporting"]["max_false_reports"]
        counter["world_false_reports"]+=row["world_false_reports"]
    return {key:dict(value) for key,value in counts.items()}


def nominal_counts(rows,policies):
    result=[]
    for policy in policies:
        subset=[r for r in rows if r["policy"]==policy and r["link"]=="good" and r["energy_condition"]=="ample"]
        base.require(len(subset)==12,"Nominal subset must retain six clips times two task deadlines")
        result.append(dict(policy=policy,runs=12,successes=sum(r["local_inspection_success"] for r in subset)))
    return result


def evidence_union(native,plan,hashes):
    path=REPO/"docs/critic_loop/critic_review_12_evidence_union.json"
    document=base.read(path);bind(path,hashes)
    bind(Path(native)/"completion.json",hashes,document["source_completion_sha256"])
    base.require({row["case_id"] for row in document["cases"]}==set(plan["cases"]),"Union diagnostic case set differs")
    summary=[]
    for case in document["cases"]:
        score_path=Path(native)/"cases"/case["case_id"]/"score.json"
        bind(score_path,hashes,case["native_score_sha256"])
        score=base.read(score_path)
        targets=plan["cases"][case["case_id"]]["evaluator_target_ids"]
        for name,scope in case["scopes"].items():
            observed={target:set() for target in targets}
            for frame in score["frames"]:
                for model in scope["models"]:
                    for target in frame["scores"][model]["matched_target_ids"]:
                        if target in observed:observed[target].add(frame["capture_time_s"])
            observed={key:sorted(values) for key,values in observed.items()}
            repeated=sorted(key for key,values in observed.items() if len(values)>=2 and values[-1]-values[0]>=2.)
            base.require(scope["matched_capture_times_s"]==observed and scope["repeated_count"]==len(repeated)
                         and sorted(scope["repeated_targets"])==repeated
                         and scope["two_of_three_repeated_evidence_possible"]==(len(repeated)>=2),
                         "Union diagnostic differs from native matched-target evidence")
        summary.append(dict(case_id=case["case_id"],
            all_native_repeated=case["scopes"]["all_native"]["repeated_count"],
            fast_only_repeated=case["scopes"]["fast_native_without_maskrcnn"]["repeated_count"],
            all_native_sensing_possible=case["scopes"]["all_native"]["two_of_three_repeated_evidence_possible"],
            fast_only_sensing_possible=case["scopes"]["fast_native_without_maskrcnn"]["two_of_three_repeated_evidence_possible"]))
    return dict(source=document,summary=summary,
        report_reconstruction="All diagnostic source unions independently reconstructed from each bound native score table.")


def summary_figure(data,output,name,title,labels):
    policies=data["policies"];fig,axes=plt.subplots(1,4,figsize=(20,9.5),sharey=True)
    fig.subplots_adjust(left=.22,right=.985,top=.81,bottom=.17,wspace=.12)
    modes=("nominal","overall","ample","residual_250j")
    titles=("Good link + ample\n12 paired conditions","All selected conditions\n72 paired conditions",
            "Ample · all links\n36 paired conditions","Residual 250 J · all links\n36 paired conditions")
    colors=("#2b8572","#596a91",base.ENERGY_COLORS["ample"],base.ENERGY_COLORS["residual_250j"])
    y=np.arange(len(policies))
    for ax,mode,title2,color in zip(axes,modes,titles,colors):
        aggregates=[]
        for policy in policies:
            if mode=="nominal":row=next(r for r in data["nominal"] if r["policy"]==policy)
            elif mode=="overall":
                subset=[r for r in data["aggregate"] if r["policy"]==policy]
                row=dict(successes=sum(r["successes"] for r in subset),runs=sum(r["runs"] for r in subset))
            else:row=next(r for r in data["aggregate"] if r["policy"]==policy and r["energy_condition"]==mode)
            aggregates.append(row)
        bars=ax.barh(y,[100*r["successes"]/r["runs"] for r in aggregates],color=color,height=.66)
        for bar,row in zip(bars,aggregates):
            ax.text(bar.get_width()+1.4,bar.get_y()+bar.get_height()/2,f'{row["successes"]}/{row["runs"]}',va="center",fontsize=8.5)
        ax.set_xlim(0,120);ax.set_xticks([0,50,100]);ax.set_xlabel("Local-task success (%)")
        ax.set_yticks(y,[labels[p] for p in policies]);ax.set_title(title2,fontsize=11,pad=12)
        ax.xaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle(title,x=.22,ha="left",fontsize=18,weight="bold")
    registration_note=("\nRemote-YOLO supplement registered after native inference began, before designer outcome inspection (41 frame journals / 2 score files already existed)."
        if "supplement_registration" in data else "")
    base.export(fig,output,name,
        "Every policy receives the same six fixed-path inspection clips per condition. All successes and failures are retained.\n"
        "Overall is an equally weighted summary of the selected constraint grid, not a deployment success rate or 72 independent flights.\n"
        "Good + ample includes both 2 s and 8 s per-invocation deadlines; each individual condition has only six correlated known-map recordings."+registration_note)


def ablation_figure(data,output):
    policies=("previous_calibrated_evidence_rate","calibrated_evidence_rate","calibrated_quality","mission_risk_budget")
    fig,axes=plt.subplots(1,2,figsize=(16.5,5.6),sharey=True)
    fig.subplots_adjust(left=.27,right=.97,top=.78,bottom=.26,wspace=.12)
    columns=[(link,budget) for link in data["links"] for budget in data["budgets"]]
    for ax,energy in zip(axes,data["energies"]):
        values=np.array([[next(r["successes"] for r in data["success_cells"] if r["policy"]==p
            and r["energy_condition"]==energy and r["link"]==link and r["perception_budget_s"]==budget)
            for link,budget in columns] for p in policies])
        ax.imshow(values/6,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f"{v}/6",ha="center",va="center",color="white" if v>=4 else "#172b3a",fontsize=11)
        ax.set_yticks(range(4),[LABELS[p] for p in policies])
        ax.set_xticks(range(6),[f"{base.LINK_LABELS[l]}\n{b} s" for l,b in columns]);ax.set_title(base.ENERGY_LABELS[energy],fontsize=12,pad=12)
        ax.axhline(1.5,color="white",lw=3)
    fig.suptitle("Same-clip ablations: training data and expected phase risk",x=.27,ha="left",fontsize=16,weight="bold")
    base.export(fig,output,"03_calibration_ablation",
        "Top pair: identical evidence-rate score and guard; only priors change from 54 to 162 development captures.\n"
        "Bottom pair: identical 162-frame quality ranking and resource guard; mission-risk adds the declared expected full-phase false-component cap.\n"
        "All comparisons use the same six new recordings. The risk gate is an expectation heuristic, not a probabilistic guarantee or correctness feedback.")


def endpoint_figure(data,output):
    fig,ax=plt.subplots(figsize=(14,9));fig.subplots_adjust(left=.32,right=.95,top=.85,bottom=.18)
    y=np.arange(len(data["policies"]))
    for offset,field,color,label in ((-.18,"original_success","#146b8c","Primary strict contract"),
        (.18,"report_centered_sensitivity_success","#bd7951","Registered endpoint sensitivity")):
        values=[sum(r[field] for r in data["endpoint_rows"] if r["policy"]==p) for p in data["policies"]]
        bars=ax.barh(y+offset,[100*v/72 for v in values],height=.32,color=color,label=label)
        for bar,value in zip(bars,values):ax.text(bar.get_width()+1.2,bar.get_y()+bar.get_height()/2,f"{value}/72",va="center",fontsize=8.5)
    ax.set_yticks(y,[LABELS[p] for p in data["policies"]]);ax.invert_yaxis();ax.set_xlim(0,115)
    ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel("Success under each evaluation endpoint (%)")
    ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.01),fontsize=9,ncol=2)
    fig.suptitle("Endpoint sensitivity does not replace the primary result",x=.32,y=.97,ha="left",fontsize=16,weight="bold")
    base.export(fig,output,"04_endpoint_sensitivity",
        "Sensitivity removes only the raw-component cap in sector evidence, keeping repeated evidence, timely correct reports, report error caps and resources.\n"
        "All 864 ORIGINAL-intent executions are rescored with the registered function; policies are not rerun and no new-intent performance is claimed.\n"
        "Primary strict outcomes remain unchanged. These are selected-grid summaries, not independent deployment trials.")


def rename_plot(output,old,new):
    for suffix in (".png",".svg"):(output/(old+suffix)).replace(output/(new+suffix))


def policy_table():
    return [
        ("Fixed local YOLO","Always local YOLO; resource failures remain part of its outcome."),
        ("Fixed local Mask R-CNN","Always local Mask; can exceed the 2 s task deadline."),
        ("Fixed local CABiNet","Always local CABiNet."),
        ("Fixed remote Mask R-CNN","Always offload Mask; does not avoid unavailable links."),
        ("Deadline-aware (old heads)","Historical RGB-based quality/risk heads; evidence-rate utility and common resource guard."),
        ("Accuracy-latency (old heads)","Historical RGB-based recall ranking under the same resource guard."),
        ("Fastest feasible","Choose the earliest forecast completion among resource-feasible actions."),
        ("Calibrated quality (162 frames)","Rank feasible actions by pooled development recall."),
        ("Evidence rate (162 frames)","Rank recall^required_hits / evidence_window, with a per-capture risk penalty."),
        ("Fixed remote YOLO","Always offload the identical YOLO source; same-model placement control."),
        ("Previous evidence rate (54 frames)","The original 54-frame prior and score, evaluated unchanged on round-two clips."),
        ("Mission risk budget (162 frames)","Filter candidates whose expected full-phase false-component count exceeds intent allowance; then rank recall."),
    ]


def html_report(data,first,plan,output):
    rows=[]
    for p in data["policies"]:
        nominal=next(r for r in data["nominal"] if r["policy"]==p)
        groups=[next(r for r in data["aggregate"] if r["policy"]==p and r["energy_condition"]==e) for e in data["energies"]]
        sensitivity=sum(r["report_centered_sensitivity_success"] for r in data["endpoint_rows"] if r["policy"]==p)
        rows.append([LABELS[p],f'{nominal["successes"]}/12',f'{sum(r["successes"] for r in groups)}/72',
            *[f'{r["successes"]}/36' for r in groups],f"{sensitivity}/72"])
    model_rows=[]
    for model in base.MODELS:
        local=next(a for a in plan["actions"] if a["source_config_id"]==model and a["placement"]=="local")
        remote=next(a for a in plan["actions"] if a["source_config_id"]==model and a["placement"]=="remote")
        model_rows.append([base.MODEL_LABELS[model],f'{local["latency_s"]:.4f} s',f'{remote["latency_s"]:.4f} s',
                           f'{local["energy_j"]:.3f} J',f'{local["load_latency_s"]:.2f} s / {local["load_energy_j"]:g} J'])
    groups=data["trajectory_comparison"]["identical_trajectory_groups"]
    group_text="; ".join(" = ".join(LABELS[p] for p in group) for group in groups) if groups else "No distinct protocol pair has identical trajectories across all 72 conditions."
    native_passes={model:sum(row["raw_phase_pass"] for row in data["native_evidence"] if row["model"]==model)
                   for model in base.MODELS}
    native_text="; ".join(f"{base.MODEL_LABELS[model]} {count}/6" for model,count in native_passes.items())
    all_model_failures=[case for case in data["case_ids"] if not any(row["raw_phase_pass"] for row in data["native_evidence"]
                       if row["case_id"]==case and row["model"] in base.MODELS)]
    union=data["evidence_union"]["summary"]
    impossible=[row["case_id"] for row in union if not row["all_native_sensing_possible"]]
    fast_possible=[row["case_id"] for row in union if row["fast_only_sensing_possible"]]
    ablations=[]
    for a,b in (("previous_calibrated_evidence_rate","calibrated_evidence_rate"),("calibrated_quality","mission_risk_budget")):
        pair=next(r for r in data["trajectory_comparison"]["pairs"] if {r["policy_a"],r["policy_b"]}=={a,b})
        gain=pair["b_only_success"] if pair["policy_b"]==b else pair["a_only_success"]
        loss=pair["a_only_success"] if pair["policy_a"]==a else pair["b_only_success"]
        ablations.append([LABELS[a]+" → "+LABELS[b],gain,loss,pair["both_success"],pair["both_failure"],pair["identical_capture_action_admission_traces"]])
    parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Perception routing: two prospective rounds</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#172b3a;background:#f1f5f7;margin:0}main{max-width:1260px;margin:auto;padding:35px 24px 65px}h1{font-size:34px;line-height:1.18}h2{font-size:23px;margin-top:35px}h3{font-size:18px}p{max-width:1080px}.scope{border-left:4px solid #146b8c;background:#e2edf2;padding:14px 18px}.note{border-left:4px solid #c18b45;background:#fff5e8;padding:14px 18px}.card{background:white;border:1px solid #dce4e9;border-radius:10px;padding:18px;margin:24px 0}img{width:100%;height:auto}a{color:#096789}.muted{color:#546778}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dbe3e8;padding:9px;text-align:left;vertical-align:top}th{background:#e7eef2}.scroll{overflow:auto}li{margin:8px 0}footer{font-size:13px;color:#526678;margin-top:35px}</style><main>
<p class="muted">30 October 2026 · Complete prospective development/evaluation record</p><h1>Perception routing on known-region, fixed inspection paths</h1>
<p class="scope"><strong>Primary second round:</strong> six new physical clips × twelve protocols × three virtual networks × two per-call deadlines × two energy envelopes = <strong>864 paired replays</strong>. All first-round 720 results are retained separately as negative transfer evidence. These are local inspection tasks on known Lake assets, not full eight-sector missions or real deployments.</p>
<p>Geometry uses an authored region of interest to ensure a meaningful inspection view. Every protocol receives the identical recorded flight, RGB frames and intent in its paired condition. The benchmark evaluates perception model and placement decisions; it does not evaluate discovering unknown target regions or changing the route.</p>
<p><strong>Fixed intent:</strong> the mission-success contract is unchanged across this comparison. The varied factors are per-call deadline, network and residual energy. The new policies use global development priors, not RGB-conditioned scene-quality predictions or online correctness feedback. The expected-risk screen consults the fixed public intent cap. This study does not establish intent generalization or scene-adaptive model selection.</p>
<h2>Second-round strict results and separate endpoint sensitivity</h2>''']
    parts.append(base.html_table(["Protocol","Good + ample","All selected conditions","Ample, all links","250 J, all links","Sensitivity only"],rows))
    parts.append('<p><strong>Denominators:</strong> good + ample is 6 clips × 2 deadlines = 12 conditions. Each energy column has 36 conditions. Overall has 72 equally weighted selected-grid conditions per policy; it is not a deployment success rate or 72 independent flights. Every individual network/deadline/energy cell contains the same six clips.</p>')
    parts.append('<p class="note"><strong>Model-pool limitation:</strong> before service costs, individual-model raw passes are '+html.escape(native_text)+'. All individual models fail in '+html.escape(", ".join(all_model_failures) or "no clip")+'. Every such clip remains included. A costly model can have useful evidence when a fast model does not; removing network access under a 2 s task deadline can then make the available useful model infeasible. These results should not be presented as broad semantic complementarity among fast and heavy models.</p>')
    parts.append('<p><strong>Generous sensing diagnostic:</strong> even a cost-free, false-positive-free union of all native models cannot provide two repeatedly observed people in '+html.escape(", ".join(impossible) or "any excluded case")+'. The fast-only union remains potentially feasible in '+html.escape(", ".join(fast_possible))+f' ({len(fast_possible)}/6 clips), so not every 2 s/offline failure is proven unavoidable. A passing union ignores one-model-per-capture selection, serialization, false positives, reporting and resources; it is neither an executable oracle nor a sufficient mission-feasibility test. The full six-clip denominator is retained.</p>')
    parts.append('<p class="note"><strong>Identical observed strategies:</strong> '+html.escape(group_text)+'. Identical trajectories are not separate empirical improvements merely because their routing objectives have different names.</p>')
    parts.append('<p><a href="plot_data.json">Exact plot data</a> · <a href="round2_run_table.json">All 864 primary rows</a> · <a href="round1_run_table.json">All 720 earlier rows</a> · <a href="endpoint_rows.json">Registered endpoint rescoring</a> · <a href="source_hashes.json">Source hashes</a> · <a href="README.txt">Reproduction</a></p>')
    figures=[("01_round2_success","Nominal and complete stress-grid results"),
        ("02_constraint_matrix","Every primary condition: six clips per cell"),
        ("03_calibration_ablation","Same-recording calibration and expected-risk ablations"),
        ("04_endpoint_sensitivity","Secondary endpoint sensitivity"),
        ("05_onboard_energy","Mean onboard nonflight energy over every run"),
        ("06_actual_choices","Actual model and placement choices"),
        ("07_native_evidence","Native repeated evidence before service costs"),
        ("08_round1_negative_context","Earlier prospective round: retained negative transfer evidence")]
    for filename,title in figures:
        parts.append(f'<section class="card"><h2>{title}</h2><a href="{filename}.png">PNG</a> · <a href="{filename}.svg">Editable SVG</a><img src="{filename}.png" alt="{title}"></section>')
    parts.append('<h2>What changed, on exactly the same second-round clips?</h2>'+base.html_table(
        ["Paired comparison","Gained strict successes","Lost strict successes","Both pass","Both fail","Identical trajectories / 72"],ablations))
    parts.append('<p>The 54→162 comparison keeps the evidence-rate rule and resource guard unchanged and changes only the complete development sample. The quality→mission-risk comparison adds only the predeclared expected full-phase false-component filter. These descriptive paired counts have no p-values or independent-trial confidence intervals.</p>')
    quality=data["error_outcomes"]["calibrated_quality"];risk=data["error_outcomes"]["mission_risk_budget"]
    parts.append('<p><strong>Risk-screen outcome:</strong> mission risk has the same 35/72 strict-success vector as pooled quality and evidence rate. '
        f'Compared with pooled quality, raw false-component-cap violating runs fall from {quality["raw_component_cap_violating_runs"]} to {risk["raw_component_cap_violating_runs"]}, '
        f'and false-report-cap violating runs from {quality["false_report_cap_violating_runs"]} to {risk["false_report_cap_violating_runs"]}. '
        'This is an error reduction on the evaluated clips, with no additional mission-success gain.</p>')
    parts.append('<h3>Optimistic evidence union, evaluator-only diagnostic</h3>'+base.html_table(
        ["Clip","All native models: repeated people","Fast models only: repeated people"],
        [[r["case_id"],f'{r["all_native_repeated"]}/3',f'{r["fast_only_repeated"]}/3'] for r in union]))
    risk_rows=[[action,*[count[k] for k in ("candidate_assessments","resource_feasible_assessments","expected_risk_exclusions_after_resource_guard","selected_decisions")]] for action,count in sorted(data["mission_risk_diagnostics"].items())]
    parts.append('<h3>Expected-risk gate audit</h3>'+base.html_table(["Candidate action","Assessments","Resource feasible","Then excluded by expected risk","Selected"],risk_rows))
    parts.append('<p>The gate forecasts <code>ceil(full phase duration / candidate capture cadence) × development false components per capture</code> and compares it with the public intent cap. This is an expected-count heuristic with no probabilistic guarantee and no observed-correctness feedback. It does not change the primary evaluator.</p>')
    parts.append('<h2>Model, service and energy assumptions</h2>'+base.html_table(
        ["Native model","Frozen local CPU service","Configured remote service","Configured local service energy","Local load cost"],model_rows))
    parts.append('<p>Models run genuine local native inference to create the mask bank. Replay uses the frozen service profile above, not the latest inference wall times. Remote service is modeled as local service time divided by four, plus virtual transport, preparation and decoding. This is not a measured remote server. Onboard energy includes configured service, load, routing, codec and radio costs; remote-server energy is excluded.</p>')
    parts.append('<p>Each physical clip contains a 36 s motion window with 18 captures and a 42 s recorded flight. Flight power is configured at 120 W, battery capacity at 120 Wh and reserve at 20%. The stress condition starts with reserve plus the 42 s flight allocation plus 250 J for task work. This is a constructed residual-energy envelope, not a measured battery-management or return-to-home model.</p>')
    parts.append('<h2>Protocol definitions</h2>'+base.html_table(["Protocol","Decision rule"],policy_table()))
    parts.append('<h2>Sequential development and evaluation boundaries</h2><ol><li>The historical 54 oblique frames trained the first global priors. First prospective sector-2/6 clips exposed transfer failure; all 648 primary and 72 supplemental replays remain reported.</li><li>All 54 historical plus all 108 first-prospective frames were promoted to development. No case, failed capture, model or threshold was filtered; the new priors use all 162 captures.</li><li>The second-round sector-3/7 recordings, twelve-policy grid, expected-risk rule and secondary endpoint were locked before second-round native inference. Exact RGB/source overlap checks find no shared training/evaluation frames. Assets and layouts remain known.</li><li>Both rounds end with complete reporting regardless of performance. No second-round result is used to alter a policy, threshold, route, site or grid.</li></ol>')
    parts.append('<p><strong>First-round supplemental timing:</strong> the fixed remote-YOLO control was registered after native inference had started, before the study designer inspected new native scores or paid outcomes. At registration, 41 frame journals and two score files already existed. It is a separate supplement; the 648-run primary plan was unchanged.</p>')
    parts.append('<h2>Endpoint and validity limits</h2><ul><li>Strict primary success retains the repeated-evidence requirement, raw false-component cap, timely focal reports, false/duplicate-report limits, execution and all resource checks.</li><li>The separately registered sensitivity removes only the raw-component part of sector evidence. It rescores original-intent executions without rerunning policies. It is not a new public-intent experiment or replacement success criterion.</li><li>The motivating first-round GT diagnostic found five unmatched fragments from true-person pixels under partial occlusion. All six GT controls reported all three focal people, but the raw component cap failed in one case. This evaluator limitation remains disclosed.</li><li>Networks are synthetic constant good/degraded/offline regimes. Paired replays are correlated conditions on six clips, not independent deployment samples. There is no unseen-map, real-flight, real-network, whole-battery-cycle or full-mission validation claim.</li></ul>')
    parts.append('<footer>Figures are generated only after both studies complete, from hash-verified artifacts. Every failure and registered condition is retained. English PNG/SVG figures and machine-readable source tables accompany this report.</footer></main></html>')
    (output/"index.html").write_text("\n".join(parts))


def build(source=ROUND2,native=NATIVE2,output=OUTPUT):
    source,native,output=(Path(p).resolve() for p in (source,native,output))
    # This is deliberately the first data read: no partial second-round outcomes.
    plan,rows,native_rows,traces,hashes=base.load_verified(source,native,864,12)
    base.require(plan["endpoint_sensitivity"]["registered_before_new_inference"] is True and
                 plan["endpoint_sensitivity"]["rule"]==RULE,"Secondary endpoint differs from pre-inference registration")
    bind(source/plan["previous_calibration_path"],hashes,plan["previous_calibration_sha256"])
    bind(source/"split_integrity.json",hashes,plan["split_integrity_sha256"])
    split=base.read(source/"split_integrity.json")
    base.require(split["passed"] and len(split["training_frames"])==162 and len(split["evaluation_frames"])==108
                 and not split["shared_rgb_sha256"] and not split["shared_source_sha256"],"Training/evaluation split failed")
    data=base.summarize(plan,rows,native_rows,traces)
    data["denominator_note"]="Each policy-energy bar contains 36 paired replay conditions from six clips; every matrix cell contains six clips. All 864 replays are correlated conditions, not independent physical missions."
    data.update(nominal=nominal_counts(rows,data["policies"]),
        trajectory_comparison=equivalence_and_ablation(rows,traces,data["policies"]),
        mission_risk_diagnostics=risk_diagnostics(source,rows,hashes),
        error_outcomes=error_outcomes(source,rows,hashes),
        evidence_union=evidence_union(native,plan,hashes),
        split_integrity=dict(passed=True,training_frames=162,evaluation_frames=108,shared_rgb_count=0,scope=split["scope"]))
    rescore,endpoint_plan=registered_rescore(hashes)
    data["endpoint_rows"]=endpoints(source,rows,rescore,hashes)

    first_plan,first_rows,first_native,first_traces,first_hashes=base.load_verified(base.DEFAULT_SOURCE,base.DEFAULT_NATIVE)
    extra_plan,extra_rows,extra_native,extra_traces,extra_hashes=base.load_verified(base.DEFAULT_SUPPLEMENT,base.DEFAULT_NATIVE,72,1)
    hashes.update(first_hashes);hashes.update(extra_hashes)
    base.require(plan["previous_round_plan_sha256"]==base.sha(base.DEFAULT_SOURCE/"plan.json")
        and extra_plan["primary_plan_sha256"]==base.sha(base.DEFAULT_SOURCE/"plan.json"),"Earlier round linkage changed")
    first=base.summarize(first_plan,first_rows,first_native,first_traces)
    extra=base.summarize(extra_plan,extra_rows,extra_native,extra_traces)
    first["policies"]+=extra["policies"]
    for key in ("aggregate","success_cells","action_choices"):first[key]+=extra[key]
    first["nominal"]=nominal_counts(first_rows+extra_rows,first["policies"])
    first["supplement_registration"]=extra_plan["registration"]
    first_endpoint=endpoints(base.DEFAULT_SOURCE,first_rows,rescore,hashes)+endpoints(base.DEFAULT_SUPPLEMENT,extra_rows,rescore,hashes)
    previous=base.read(ENDPOINT1/"results.json");bind(ENDPOINT1/"results.json",hashes)
    base.require(previous["complete"] and previous["runs"]==720 and previous["plan_sha256"]==base.sha(ENDPOINT1/"plan.json"),"First endpoint result is incomplete")
    previous_by_id={row["run_id"]:row for row in previous["rows"]}
    base.require(len(previous_by_id)==720,"Missing first endpoint rows")
    for row in first_endpoint:
        old=previous_by_id[row["run_id"]]
        base.require(all(old[k]==row[k] for k in ("original_success","report_centered_sensitivity_success","raw_component_cap_only_failure","checks")),"First sensitivity differs from registered result")
    first["endpoint_rows"]=first_endpoint
    data["first_round"]=first
    data["protocol_definitions"]=policy_table()
    data["profile_assumptions"]=dict(actions=plan["actions"],network=plan["network"],battery_design=plan["battery_design"],policy_costs=plan["policy_costs"])
    data["scope_limits"]=dict(intent="Fixed mission-success contract; no intent-generalization experiment",
        varied_factors=["per-call deadline","network","residual energy"],
        new_policy_inputs="Global development quality/risk priors plus causal public resource state; no scene-conditioned quality head or online correctness feedback",
        expected_risk="Uses the fixed public false-component allowance, not an observed-error ledger",
        task="Known-region local inspection with fixed paths on known Lake assets; not unknown-region search or full eight-sector missions")
    output.mkdir(parents=True,exist_ok=True)
    base.save(output/"plot_data.json",data)
    base.save(output/"round2_run_table.json",rows)
    base.save(output/"round1_run_table.json",first_rows+extra_rows)
    base.save(output/"endpoint_rows.json",dict(round2=data["endpoint_rows"],round1=first_endpoint,
        round2_rule=plan["endpoint_sensitivity"],round1_registration=endpoint_plan["registration"]))
    base.LABELS.update(LABELS);base.style()
    summary_figure(data,output,"01_round2_success","Second prospective round: nominal and stress-grid success",LABELS)
    base.matrix(data,output)
    ablation_figure(data,output)
    endpoint_figure(data,output)
    base.energy_figure(data,output);rename_plot(output,"03_onboard_energy","05_onboard_energy")
    base.choice_figure(data,output);rename_plot(output,"05_actual_choices","06_actual_choices")
    base.native_table(data,output);rename_plot(output,"04_native_evidence","07_native_evidence")
    first_labels={**LABELS,"calibrated_quality":"Calibrated quality (54 frames)",
        "calibrated_evidence_rate":"Evidence rate (54 frames)","fixed_remote_yolo":"Remote YOLO (separate supplement)"}
    summary_figure(first,output,"08_round1_negative_context","First prospective round: prior-transfer failures retained",first_labels)
    html_report(data,first,plan,output)
    bind(__file__,hashes);bind(base.__file__,hashes)
    base.save(output/"source_hashes.json",dict(schema_version="two_round_routing_report_sources_v1",sha256=hashes))
    shutil.copyfile(__file__,output/"consolidated_report_builder.py")
    shutil.copyfile(base.__file__,output/"first_report_helpers.py")
    (output/"README.txt").write_text("Two prospective routing rounds: English figures and exact data\n\n"
        "Run from the repository root:\nPYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/strong-routing-mpl /usr/bin/python3 "
        "-m experiments.strong_routing_20261030.consolidated_report\n\n"
        "Requires all864 second-round runs, all648 first-round primary runs, all72 first-round supplemental runs, "
        "complete native cohorts and registered first-round endpoint results. Partial second-round data are refused before plotting.\n"
        "The exact registered rescore function is loaded from its hash-verified AST in an isolated namespace with its validation helper. "
        "Only the evaluator endpoint changes; original public-intent executions and strict outcomes remain unchanged.\n"
        "Each policy has six physical clips evaluated across72 selected paired conditions. These are not72 independent flights or a deployment-rate estimate. "
        "The good-link ample subset has12 paired conditions. Every individual matrix cell has6.\n"
        "All plots are PNG180dpi and editable SVG. Raw tables, plotting data, policy equivalence, model choices, endpoint results and hashes are included. "
        "Builder source copies accompany the report; reproduction imports the repository helper module. No PDF is required.\n")
    base.save(output/"build_receipt.json",dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        round2_runs=len(rows),round1_runs=len(first_rows)+len(extra_rows),round2_physical_clips=6,source_hash_count=len(hashes),
        artifact_sha256={str(p.relative_to(output)):base.sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="build_receipt.json"}))
    print(output/"index.html")
    return data


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROUND2)
    parser.add_argument("--native",type=Path,default=NATIVE2)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    build(args.source,args.native,args.output)
