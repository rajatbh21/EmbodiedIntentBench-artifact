"""Render only completed, hash-bound prospective routing comparison results.

Usage:
  PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/strong-routing-mpl /usr/bin/python3 \
    -m experiments.strong_routing_20261030.report

This module uses the standard library, NumPy and Matplotlib only. It never
imports, modifies, reruns, fits or selects an experimental policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
from itertools import combinations
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "results/strong_routing_comparison_v2_20261030"
DEFAULT_NATIVE = REPO / "results/strong_routing_native_v1_20261030"
DEFAULT_OUTPUT = REPO / "reports/strong_routing_comparison_20261030"
DEFAULT_SUPPLEMENT = REPO / "results/strong_routing_remote_yolo_control_v1_20261030"
DEFAULT_GEOMETRY = REPO / "results/strong_routing_geometry_controls_v1_20261030"
LABELS = {
    "fixed_local_yolo": "Fixed local YOLO",
    "fixed_local_mask": "Fixed local Mask R-CNN",
    "fixed_local_cabinet": "Fixed local CABiNet",
    "fixed_remote_mask": "Fixed remote Mask R-CNN",
    "deadline_aware": "Deadline-aware (old heads)",
    "accuracy_latency": "Accuracy-latency (old heads)",
    "fastest_feasible": "Fastest feasible",
    "calibrated_quality": "Calibrated quality",
    "calibrated_evidence_rate": "Calibrated evidence rate",
    "fixed_remote_yolo": "Fixed remote YOLO (supplement)",
}
MODELS = ("CFG_YOLO11S_960_ROT0", "CFG_MASKRCNN_1280", "CFG_CABINET_UAVID_1280")
MODEL_LABELS = dict(zip(MODELS, ("YOLO11s", "Mask R-CNN", "CABiNet")))
ENERGY_LABELS = {"ample": "Ample battery", "residual_250j": "250 J residual task allowance"}
ENERGY_COLORS = {"ample": "#146b8c", "residual_250j": "#e39935"}
LINK_LABELS = {"good": "Good", "degraded": "Degraded", "offline": "Offline"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, document):
    Path(path).write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n")


def close(a, b):
    return abs(a-b) <= 1e-8 * max(1., abs(a), abs(b))


def load_verified(source, native, expected_runs=648, expected_policies=9):
    """Refuse partial data; verify every row artifact and every plotted count."""
    source, native = Path(source).resolve(), Path(native).resolve()
    require((source / "completion.json").is_file(), "Paid study is still incomplete; no report was generated")
    require((native / "completion.json").is_file(), "Native study is still incomplete; no report was generated")
    hashes = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = hashes.get(str(path))
        if actual is None:
            actual = sha(path)
            hashes[str(path)] = actual
        require(expected is None or actual == expected, "Changed source artifact: " + str(path))
        return actual

    plan, complete = read(source / "plan.json"), read(source / "completion.json")
    require(complete["complete"] and complete["runs"] == plan["expected_runs"] == expected_runs,
            f"All {expected_runs} declared paid runs are required")
    bind(source / "completion.json")
    bind(source / "plan.json", complete["plan_sha256"])
    bind(source / "run_index.json", complete["run_index_sha256"])
    bind(source / "native_bindings.json", complete["native_bindings_sha256"])
    bind(source / plan["calibration_path"], plan["calibration_sha256"])
    for name, path in plan["profiles"].items():
        bind(source / path, plan["profile_sha256"][name])
    for path, expected in plan["source_hashes"].items():
        bind(path, expected)
    for case, expected in complete["prediction_manifest_sha256"].items():
        bind(source / "predictions" / case / "manifest.json", expected)
    rows = read(source / "run_index.json")
    declared = {row["run_id"]: row for row in plan["grid"]}
    require(len(rows) == len(declared) == len({row["run_id"] for row in rows}) == expected_runs,
            "Missing or duplicate replay rows")
    require({row["run_id"] for row in rows} == set(declared), "Replay grid differs from its frozen plan")
    require(len(plan["cases"]) == 6 and len(plan["factors"]["policies"]) == expected_policies,
            f"Six cases and {expected_policies} policies required")
    traces = {}
    for row in rows:
        require(all(row.get(k) == v for k, v in declared[row["run_id"]].items()), "Changed replay factors")
        folder = source / row["path"]
        for name, expected in row["artifacts_sha256"].items():
            bind(folder / name, expected)
        outcome, result, events = (read(folder / name) for name in ("outcome.json", "results.json", "events.json"))
        require(row["local_inspection_success"] == outcome["local_inspection_success"]
                == all(outcome["checks"].values()), "Success predicate differs from outcome")
        require(row["full_wide_mission_evaluated"] is False, "Local clips must not be relabeled full missions")
        require(close(row["nonflight_energy_j"], result["cumulative_nonflight_energy_j"]), "Changed energy total")
        counts = Counter(event["selected_action_id"] for event in events if event["selected_action_id"] is not None)
        require(dict(counts) == row["selected_action_counts"], "Action counts differ from event receipts")
        require(row["abstentions"] == sum(event["selected_action_id"] is None for event in events)
                and row["decision_count"] == len(events), "Changed decision denominator")
        row["failure_checks_for_report"] = outcome["failed_constraints"]
        traces[row["run_id"]] = [(event["observation_id"], event["selected_action_id"],
            event["perception"]["admitted"], event["perception"]["status"]) for event in events]
    require(sum(row["local_inspection_success"] for row in rows) == complete["local_inspection_successes"],
            "Completion success count differs from outcomes")

    native_plan, native_completion = read(native / "protocol.json"), read(native / "completion.json")
    bind(native / "protocol.json", plan["native_protocol_sha256"])
    require(native_completion["complete"] and native_completion["protocol_sha256"] == sha(native / "protocol.json"),
            "Complete bound native study required")
    bind(native / "completion.json")
    bind(native / "acquisition_bindings.json", native_completion["acquisition_bindings_sha256"])
    require({row["case_id"] for row in native_completion["cases"]} == set(plan["cases"]),
            "Native and paid case sets differ")
    native_rows = []
    for row in native_completion["cases"]:
        score_path = native / "cases" / row["case_id"] / "score.json"
        bind(score_path, row["score_sha256"])
        score = read(score_path)
        require(row["status"] == "completed", "A native case is incomplete")
        for path, expected in score["input_sha256"].items():
            bind(path, expected)
        require(set(score["models"]) == set(MODELS) | {"perfect_gt"}, "Wrong native model pool")
        for model in (*MODELS, "perfect_gt"):
            measured = score["models"][model]
            focal = measured["focal_sector"]
            require(focal["target_count"] == 3 and measured["raw_phase_pass"]
                    == focal["zero_cost_phase_requirement_met"], "Native phase outcome differs")
            native_rows.append(dict(case_id=row["case_id"], model=model,
                supported_targets=focal["supported_target_count"], focal_targets=3,
                false_components=focal["false_components"], raw_phase_pass=measured["raw_phase_pass"],
                min_hits=focal["min_hits"], min_evidence_span_s=focal["min_evidence_span_s"],
                max_false_components=focal["max_false_components"]))
    return plan, rows, native_rows, traces, hashes


def summarize(plan, rows, native_rows, traces):
    policies, links, energies = (plan["factors"][k] for k in ("policies", "links", "energy_conditions"))
    budgets = plan["factors"]["perception_budget_s"]
    groups, cells, choices = [], [], []
    for policy in policies:
        for energy in energies:
            subset = [r for r in rows if r["policy"] == policy and r["energy_condition"] == energy]
            require(len(subset) == 36, "Each policy-energy aggregate must contain all 36 paired cells")
            total = Counter()
            for row in subset:
                total.update(row["selected_action_counts"])
            total["abstain"] = sum(row["abstentions"] for row in subset)
            choices.append(dict(policy=policy, energy_condition=energy, runs=len(subset),
                decision_count=sum(row["decision_count"] for row in subset), counts=dict(total),
                skipped_opportunities=sum(row["skipped_opportunity_count"] for row in subset)))
            groups.append(dict(policy=policy, energy_condition=energy, runs=len(subset),
                successes=sum(row["local_inspection_success"] for row in subset),
                mean_nonflight_energy_j=float(np.mean([row["nonflight_energy_j"] for row in subset])),
                mean_total_energy_j=float(np.mean([row["total_energy_j"] for row in subset])),
                failed_constraints=dict(Counter(k for row in subset for k in row["failure_checks_for_report"]))))
            for link in links:
                for budget in budgets:
                    cell = [r for r in subset if r["link"] == link and r["perception_budget_s"] == budget]
                    require(len(cell) == 6 and {r["flight"] for r in cell} == set(plan["cases"]),
                            "Each heatmap cell must contain all six paired clips")
                    cells.append(dict(policy=policy, energy_condition=energy, link=link,
                        perception_budget_s=budget, runs=6,
                        successes=sum(row["local_inspection_success"] for row in cell)))
    equivalences = []
    new_policies = [p for p in policies if p in ("fastest_feasible", "calibrated_quality", "calibrated_evidence_rate")]
    for a, b in combinations(new_policies, 2):
        by_key = {}
        for row in rows:
            if row["policy"] not in (a, b):
                continue
            key = (row["flight"], row["perception_budget_s"], row["link"], row["energy_condition"])
            by_key.setdefault(key, {})[row["policy"]] = row
        require(len(by_key) == 72 and all(set(pair) == {a,b} for pair in by_key.values()), "Unpaired policy comparison")
        equal_trace = sum(traces[pair[a]["run_id"]] == traces[pair[b]["run_id"]] for pair in by_key.values())
        equal_success = sum(pair[a]["local_inspection_success"] == pair[b]["local_inspection_success"] for pair in by_key.values())
        equal_energy = sum(close(pair[a]["nonflight_energy_j"], pair[b]["nonflight_energy_j"]) for pair in by_key.values())
        equivalences.append(dict(policy_a=a, policy_b=b, paired_conditions=72,
            identical_capture_action_admission_traces=equal_trace,
            identical_success_outcomes=equal_success, identical_nonflight_energy=equal_energy))
    return dict(schema_version="strong_routing_report_plot_data_v1", policies=policies, links=links,
        energies=energies, budgets=budgets, case_ids=list(plan["cases"]), aggregate=groups,
        success_cells=cells, action_choices=choices, native_evidence=native_rows,
        new_policy_equivalence=equivalences,
        denominator_note="Each policy-energy bar contains 36 paired replay conditions from six clips; every matrix cell contains six clips. Replays are correlated, not 648 independent physical missions.")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#172b3a", "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
        "savefig.facecolor": "white"})


def export(fig, output, name, note):
    fig.text(.015, .015, note, fontsize=8.4, color="#536476", va="bottom")
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def tied_new_policies(data):
    rows=data["new_policy_equivalence"]
    return len(rows)==3 and all(row["identical_capture_action_admission_traces"]==72 for row in rows)


def overview(data, output):
    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    fig.subplots_adjust(left=.27, right=.95, top=.88, bottom=.18)
    y = np.arange(len(data["policies"]))
    for offset, energy in zip((-.19, .19), data["energies"]):
        rows = [next(r for r in data["aggregate"] if r["policy"] == policy and r["energy_condition"] == energy)
                for policy in data["policies"]]
        values = [r["successes"]/r["runs"]*100 for r in rows]
        bars = ax.barh(y+offset, values, height=.34, color=ENERGY_COLORS[energy], label=ENERGY_LABELS[energy])
        for bar, row in zip(bars, rows):
            ax.text(bar.get_width()+1.1, bar.get_y()+bar.get_height()/2,
                f'{row["successes"]}/{row["runs"]}', va="center", fontsize=9)
    ax.set_yticks(y, [LABELS[p] for p in data["policies"]]); ax.invert_yaxis()
    ax.set_xlim(0, 114); ax.set_xticks([0,20,40,60,80,100]); ax.set_xlabel("Local-inspection success (%)")
    ax.set_title("Complete local-task success by routing protocol", loc="left", pad=32, fontsize=16)
    ax.legend(loc="lower left", bbox_to_anchor=(0,1.01), ncol=2, frameon=False, fontsize=9)
    ax.xaxis.grid(True, alpha=.18); ax.set_axisbelow(True)
    note=(
        "Each bar: 6 fixed clips × 3 network regimes × 2 per-call deadlines = 36 paired conditions.\n"
        "Includes every success and failure. Six prospective clips on known Lake assets; no independent-trial confidence intervals.")
    if tied_new_policies(data):
        note+="\nThe three new objectives select identical capture/action/admission trajectories in all 72 paired conditions."
    export(fig, output, "01_protocol_success",note)


def matrix(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.2), sharey=True)
    fig.subplots_adjust(left=.23, right=.93, top=.83, bottom=.18, wspace=.11)
    columns = [(link,budget) for link in data["links"] for budget in data["budgets"]]
    for ax, energy in zip(axes, data["energies"]):
        values = np.array([[next(r["successes"] for r in data["success_cells"] if r["policy"] == policy
            and r["energy_condition"] == energy and r["link"] == link and r["perception_budget_s"] == budget)
            for link,budget in columns] for policy in data["policies"]])
        artist = ax.imshow(values/6, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for (i,j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value}/6", ha="center", va="center", color="white" if value>=4 else "#172b3a", fontsize=10)
        ax.set_xticks(range(len(columns)), [f"{LINK_LABELS[l]}\n{b:g} s" for l,b in columns])
        ax.set_yticks(range(len(data["policies"])), [LABELS[p] for p in data["policies"]])
        ax.set_title(ENERGY_LABELS[energy], pad=13, fontsize=12)
        for x in (1.5,3.5): ax.axvline(x, color="white", lw=2)
    fig.suptitle("Which constraints change local-task success?", x=.23, ha="left", fontsize=16, weight="bold")
    colorbar = fig.colorbar(artist, ax=axes, fraction=.025, pad=.025, ticks=[0,.5,1])
    colorbar.ax.set_yticklabels(["0%","50%","100%"])
    export(fig, output, "02_constraint_matrix",
        "Each cell is the same six clips, with one protocol, network regime, per-call deadline and energy envelope.\n"
        "Good: 50/100 Mbps up/down, 30 ms RTT. Degraded: 2/10 Mbps, 150 ms. Offline: full-duration outage. All networks are virtual.")


def energy_figure(data, output):
    fig, axes = plt.subplots(1,2,figsize=(15.0,7.2),sharey=True)
    fig.subplots_adjust(left=.23,right=.96,top=.83,bottom=.18,wspace=.12)
    y=np.arange(len(data["policies"]))
    largest=max(r["mean_nonflight_energy_j"] for r in data["aggregate"])
    for ax,energy in zip(axes,data["energies"]):
        rows=[next(r for r in data["aggregate"] if r["policy"]==p and r["energy_condition"]==energy) for p in data["policies"]]
        bars=ax.barh(y,[r["mean_nonflight_energy_j"] for r in rows],color=ENERGY_COLORS[energy],height=.65)
        for bar,row in zip(bars,rows):
            ax.text(bar.get_width()+largest*.012,bar.get_y()+bar.get_height()/2,
                f'{row["mean_nonflight_energy_j"]:.1f} J  ·  {row["successes"]}/36',fontsize=8.5,va="center")
        ax.set_yticks(y,[LABELS[p] for p in data["policies"]]);ax.set_xlim(0,largest*1.32)
        ax.set_xlabel("Mean configured onboard nonflight energy (J)")
        ax.set_title(ENERGY_LABELS[energy],pad=13,fontsize=12)
        ax.xaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Energy is averaged over successes and failures",x=.23,ha="left",fontsize=16,weight="bold")
    export(fig,output,"03_onboard_energy",
        "Labels: mean energy and successful runs / 36. Low energy can reflect canceled work or abstention; it is not sufficient for mission success.\n"
        "Configured service, model-loading, routing, codec and radio energy; excludes flight energy and remote-server energy. No measured battery claim.")


def native_table(data,output):
    fig,ax=plt.subplots(figsize=(12.8,7.2));fig.subplots_adjust(left=.28,right=.98,top=.83,bottom=.19)
    values=np.zeros((len(data["case_ids"]),len(MODELS)))
    for i,case in enumerate(data["case_ids"]):
        for j,model in enumerate(MODELS):
            row=next(r for r in data["native_evidence"] if r["case_id"]==case and r["model"]==model)
            values[i,j]=row["raw_phase_pass"]
    ax.imshow(values,vmin=0,vmax=1,cmap=matplotlib.colors.ListedColormap(["#f5dfdc","#d5e9e2"]),aspect="auto")
    for i,case in enumerate(data["case_ids"]):
        for j,model in enumerate(MODELS):
            row=next(r for r in data["native_evidence"] if r["case_id"]==case and r["model"]==model)
            passed="PASS" if row["raw_phase_pass"] else "FAIL"
            ax.text(j,i,f'{row["supported_targets"]}/3 repeated targets\n{passed}  ·  {row["false_components"]} false components',
                    ha="center",va="center",fontsize=10)
    ax.set_yticks(range(len(data["case_ids"])),[case.replace("/", " · ").replace("_", " ") for case in data["case_ids"]])
    ax.set_xticks(range(len(MODELS)),[MODEL_LABELS[m] for m in MODELS]);ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-.5,len(MODELS),1),minor=True);ax.set_yticks(np.arange(-.5,len(data["case_ids"]),1),minor=True)
    ax.grid(which="minor",color="white",lw=2);ax.tick_params(which="minor",length=0)
    fig.suptitle("Native perception evidence before service costs",x=.28,ha="left",fontsize=16,weight="bold")
    export(fig,output,"04_native_evidence",
        "All 18 motion captures per clip; actual neural outputs. A raw pass needs ≥2 of 3 focal targets, each seen twice ≥2 s apart, and ≤2 false components.\n"
        "This table ignores service latency and resource costs. A raw pass is not a paid local-task pass or a full eight-sector mission result.")


def choice_figure(data,output):
    fig,axes=plt.subplots(1,2,figsize=(15.0,7.8),sharey=True)
    fig.subplots_adjust(left=.23,right=.97,top=.78,bottom=.20,wspace=.10)
    actions=[m for m in MODELS]+["REMOTE__"+m for m in MODELS]+["abstain"]
    colors=["#227a9d","#565a9e","#648c44","#7dc5da","#aaa6d9","#b4d08f","#b3b8be"]
    y=np.arange(len(data["policies"]))
    maxdec=max(r["decision_count"] for r in data["action_choices"])
    for ax,energy in zip(axes,data["energies"]):
        rows=[next(r for r in data["action_choices"] if r["policy"]==p and r["energy_condition"]==energy) for p in data["policies"]]
        left=np.zeros(len(rows))
        for action,color in zip(actions,colors):
            values=np.array([r["counts"].get(action,0) for r in rows])
            label="Abstain" if action=="abstain" else ("Remote " if action.startswith("REMOTE__") else "Local ")+MODEL_LABELS[action.removeprefix("REMOTE__")]
            ax.barh(y,values,left=left,height=.68,color=color,label=label)
            for i,value in enumerate(values):
                if value>=maxdec*.055:
                    ax.text(left[i]+value/2,i,str(value),ha="center",va="center",fontsize=8,color="#152c36")
            left+=values
        for i,row in enumerate(rows):
            require(left[i]==row["decision_count"],"Choice chart denominator mismatch")
            ax.text(left[i]+maxdec*.012,i,str(row["decision_count"]),va="center",fontsize=8.5)
        ax.set_yticks(y,[LABELS[p] for p in data["policies"]]);ax.set_xlim(0,maxdec*1.11)
        ax.set_xlabel("Actual routing decisions, including abstentions")
        ax.set_title(ENERGY_LABELS[energy],pad=12,fontsize=12)
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.61,.92),ncol=4,frameon=False,fontsize=8.5)
    fig.suptitle("What each protocol actually selected",x=.23,ha="left",fontsize=16,weight="bold")
    export(fig,output,"05_actual_choices",
        "Each row aggregates all 36 paired replay conditions. Right-hand labels are total decision counts; segments are actual action counts.\n"
        "Busy captures are skipped and are not routing decisions. Counts are not independent samples or evidence that a selected output was admitted.")


def supplemental_figure(data,output):
    policies=("fixed_local_yolo","fixed_remote_yolo","calibrated_quality")
    columns=[(link,budget) for link in data["links"] for budget in data["budgets"]]
    cells=data["success_cells"]+data["supplemental"]["success_cells"]
    fig,axes=plt.subplots(1,2,figsize=(15.4,4.8),sharey=True)
    fig.subplots_adjust(left=.25,right=.96,top=.73,bottom=.28,wspace=.12)
    for ax,energy in zip(axes,data["energies"]):
        values=np.array([[next(r["successes"] for r in cells if r["policy"]==policy and
            r["energy_condition"]==energy and r["link"]==link and r["perception_budget_s"]==budget)
            for link,budget in columns] for policy in policies])
        ax.imshow(values/6,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        for (i,j),value in np.ndenumerate(values):
            ax.text(j,i,f"{value}/6",ha="center",va="center",color="white" if value>=4 else "#172b3a",fontsize=11)
        ax.set_yticks(range(3),[LABELS[p] for p in policies])
        ax.set_xticks(range(len(columns)),[f"{LINK_LABELS[l]}\n{b:g} s" for l,b in columns])
        ax.set_title(ENERGY_LABELS[energy],pad=12,fontsize=12)
        for x in (1.5,3.5):ax.axvline(x,color="white",lw=2)
    fig.suptitle("Supplement: isolate model placement and feasible fallback",x=.25,ha="left",fontsize=15,weight="bold")
    export(fig,output,"06_supplemental_placement",
        "The 72 fixed-remote-YOLO runs are a separately registered supplement; primary 648 runs are unchanged. All cells contain the same six clips.\n"
        "Supplement registered after native inference started and before new score inspection by the designer. Actual registration state is disclosed in plot_data.json.\n"
        "Model identity is the same YOLO source; network, budgets and configured costs are paired. Calibrated quality is one representative new objective.")


def add_geometry(data,folder,hashes):
    folder=Path(folder).resolve()
    complete=read(folder/"completion.json")
    require(complete["complete"] and len(complete["cases"])==6,"All six GT geometry controls required")
    expected={str(folder/"plan.json"):complete["plan_sha256"],str(folder/"completion.json"):sha(folder/"completion.json")}
    for row in complete["cases"]:
        path=folder/row["artifact"]
        expected[str(path)]=row["artifact_sha256"]
        control=read(path)
        expected.update(control["input_sha256"])
        expected.update(control["implementation_sha256"])
    fragment_path=folder/"shoreline_sector2_gt_fragment_audit.json"
    fragment=read(fragment_path)
    expected[str(fragment_path)]=sha(fragment_path)
    expected.update(fragment["input_sha256"])
    for path,value in expected.items():
        actual=hashes.get(path) or sha(path)
        require(actual==value,"Changed GT diagnostic input: "+path)
        hashes[path]=actual
    data["gt_geometry_controls"]=complete
    data["gt_fragmentation_audit"]=fragment


def html_table(headers, rows):
    return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(x))}</th>" for x in headers)+"</tr></thead><tbody>"+"".join(
        "<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)+"</tbody></table>"


def write_html(data, plan, output):
    figures=[("01_protocol_success","Protocol success", "Every predeclared clip and constraint combination is retained."),
        ("02_constraint_matrix","Paired constraint matrix", "Six matched physical clips per cell. The deadline is per perception invocation."),
        ("03_onboard_energy","Onboard nonflight energy", "All runs contribute, including failures, canceled inference and abstention."),
        ("04_native_evidence","Native model evidence", "Cost-relaxed raw evidence; it must not be read as a paid task success rate."),
        ("05_actual_choices","Actual model and placement choices", "Exact counts and policy trace comparisons are included in the data download."),
        ("06_supplemental_placement","Separately registered placement control", "This 72-run supplement adds fixed remote YOLO without changing the primary 648-run plan.")]
    table_rows=[]
    for policy in data["policies"]:
        cells=[next(r for r in data["aggregate"] if r["policy"]==policy and r["energy_condition"]==e) for e in data["energies"]]
        table_rows.append([LABELS[policy],*[f'{r["successes"]}/36 ({100*r["successes"]/36:.1f}%)' for r in cells]])
    equal=[]
    for pair in data["new_policy_equivalence"]:
        equal.append(f'{LABELS[pair["policy_a"]]} vs {LABELS[pair["policy_b"]]}: '
            f'{pair["identical_capture_action_admission_traces"]}/72 identical capture/action/admission traces; '
            f'{pair["identical_success_outcomes"]}/72 identical success outcomes; '
            f'{pair["identical_nonflight_energy"]}/72 identical energy totals.')
    exact_choices=[]
    for row in data["action_choices"]:
        exact_choices.append([LABELS[row["policy"]],ENERGY_LABELS[row["energy_condition"]],row["decision_count"],
            row["skipped_opportunities"],", ".join(f'{key}: {value}' for key,value in sorted(row["counts"].items()))])
    native_gt=[r for r in data["native_evidence"] if r["model"]=="perfect_gt"]
    gt_count=sum(r["raw_phase_pass"] for r in native_gt)
    tie_notice=("<p class=\"scope\"><strong>Observed equivalence:</strong> all three new objectives choose identical capture/action/admission trajectories in all 72 paired conditions. Their success and efficiency should be treated as one observed strategy here, not three independently demonstrated improvements.</p>" if tied_new_policies(data) else "")
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prospective fixed-flight perception routing comparison</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#172b3a;background:#f2f5f7;margin:0}}main{{max-width:1180px;margin:auto;padding:36px 24px 70px}}
h1{{font-size:32px;line-height:1.18}}h2{{font-size:23px;margin-top:36px}}h3{{font-size:17px}}p{{max-width:1000px}}.muted{{color:#526678}}.scope{{border-left:4px solid #146b8c;background:#e5f0f3;padding:14px 18px}}.card{{background:white;border:1px solid #dbe3e8;border-radius:10px;padding:18px;margin:22px 0}}img{{width:100%;height:auto}}a{{color:#096789}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dbe3e8;padding:9px;text-align:left;vertical-align:top}}th{{background:#eaf0f3}}.scroll{{overflow:auto}}li{{margin:8px 0}}code{{font-size:12px;overflow-wrap:anywhere}}footer{{margin-top:32px;color:#526678;font-size:13px}}
</style><main><p class="muted">Study date: 30 October 2026 · English figure and data bundle</p>
<h1>Prospective fixed-flight perception routing comparison</h1>
<p class="scope"><strong>Scope:</strong> 6 new physical inspection clips in 3 known Lake layouts; 9 policies × 3 virtual networks × 2 per-call deadlines × 2 energy envelopes = <strong>648 paired replays</strong>. These are local inspection tasks, not full eight-sector missions or 648 independent physical flights.</p>
<h2>Measured local-task outcomes</h2>{html_table(["Protocol","Ample battery","250 J residual task allowance"],table_rows)}{tie_notice}
<p>The complete contract requires repeated evidence for at least 2 of 3 focal people and timely reports, plus false-positive, duplicate, execution and resource checks. Model weights and criteria are shared; protocols receive the same recording per condition.</p>
<p><strong>Interpretation limits:</strong> new recordings were scored after the routing plan was frozen, but use known map assets and target layouts. Service and energy costs are configured. Remote inference service is assumed to take one quarter of local inference time; the server is not measured and its energy is excluded. The 250 J residual envelope is an analytical stress condition after reserving the 42 s flight and 20% battery reserve.</p>
<p><strong>Known metric limitation:</strong> every GT-union control reports all 3 focal people, but shoreline sector 2 fails the raw false-component cap. Five unmatched components at one partly occluded person contain only true-person pixels. This is fragmentation under one-to-one component matching, not semantic pixel hallucination. The case remains included in every result.</p>
<p><a href="plot_data.json">Exact plot data</a> · <a href="run_table.json">All 648 primary replay rows</a> · <a href="supplemental_run_table.json">72 supplementary replay rows</a> · <a href="source_hashes.json">Source hashes</a> · <a href="report_builder.py">Reproducible report builder</a> · <a href="README.txt">Reproduction instructions</a></p>
'''
    for name,title,description in figures:
        content+=f'<section class="card"><h2>{title}</h2><p>{description}</p><a href="{name}.svg">Editable SVG</a> · <a href="{name}.png">PNG</a><img src="{name}.png" alt="{title}"></section>'
    content+='<h2>Do the new protocols actually differ?</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in equal)+'</ul>'
    content+='<p>Identical action traces are an observed equivalence on this finite grid. They do not support separate superiority claims for differently named objectives.</p>'
    content+=f'<h2>Validity checks and limitations</h2><ul><li>All 648 predeclared replay cells and all six native clips are present. Every plotted outcome, energy total and action count was cross-checked against hashed run artifacts.</li><li>Perfect-GT raw focal evidence passes in {gt_count}/6 clips. This validates only the stated raw evidence check, not deployment realism or paid-model success.</li><li>New global priors use all 54 old oblique development frames, including the prior failed revisit case. They are point estimates with smoothing, not statistical confidence bounds.</li><li>Every matrix cell has the same six clips. No confidence intervals, p-values, or claims of population significance are computed from repeated factorial replays.</li><li>Constant good, degraded and offline traces are synthetic. The energy envelope is not a measured BMS or return-to-home model. Actual native inference is run locally; replay charges declared service costs.</li></ul>'
    content+='<h3>GT-union diagnostics, not a deployable protocol</h3>'+html_table(
        ["Clip","Repeated targets","Focal reports","False components","Raw focal pass"],
        [[r["case_id"],f'{r["repeated_targets"]}/3',f'{r["focal_reports"]}/3',r["raw_false_components"],r["raw_sector_pass"]]
         for r in data["gt_geometry_controls"]["cases"]])
    supplemental=data["supplemental"]
    content+='<h3>Supplement registration</h3><p>'+html.escape(supplemental["registration"])+"</p><p>Native artifacts already present at registration: "+html.escape(json.dumps(supplemental["native_progress_at_registration"],sort_keys=True))+". These files existed; the registration states their new scores had not yet been inspected by the designer.</p>"
    content+=html_table(["Supplemental protocol","Energy condition","Success"],[[LABELS[r["policy"]],ENERGY_LABELS[r["energy_condition"]],f'{r["successes"]}/36'] for r in supplemental["aggregate"]])
    content+='<h2>Exact routing-choice counts</h2><div class="scroll">'+html_table(
        ["Protocol","Energy condition","Decisions","Busy captures skipped","Selected actions, including abstention"],exact_choices)+'</div>'
    content+='<footer>All figures are generated directly from completed, hash-verified artifacts. Original studies and frozen policies are unchanged.</footer></main></html>'
    (output/"index.html").write_text(content)


def build(source=DEFAULT_SOURCE,native=DEFAULT_NATIVE,output=DEFAULT_OUTPUT,
          supplement=DEFAULT_SUPPLEMENT,geometry=DEFAULT_GEOMETRY):
    source,native,output=map(lambda p:Path(p).resolve(),(source,native,output))
    plan,rows,native_rows,traces,hashes=load_verified(source,native)
    data=summarize(plan,rows,native_rows,traces)
    extra_plan,extra_rows,extra_native,extra_traces,extra_hashes=load_verified(supplement,native,72,1)
    require(extra_plan["primary_plan_sha256"]==sha(source/"plan.json"),"Supplement parent differs")
    require(extra_plan["cases"]==plan["cases"] and extra_plan["actions"]==plan["actions"],"Unpaired supplement cases or models")
    hashes.update(extra_hashes)
    extra=summarize(extra_plan,extra_rows,extra_native,extra_traces)
    extra.update(registration=extra_plan["registration"],native_progress_at_registration=extra_plan["native_progress_at_registration"])
    data["supplemental"]=extra
    add_geometry(data,geometry,hashes)
    output.mkdir(parents=True,exist_ok=True)
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    save(output/"plot_data.json",data)
    save(output/"run_table.json",rows)
    save(output/"supplemental_run_table.json",extra_rows)
    save(output/"source_hashes.json",dict(schema_version="strong_routing_report_sources_v1",sha256=hashes))
    style()
    for plot in (overview,matrix,energy_figure,native_table,choice_figure,supplemental_figure):
        plot(data,output)
    write_html(data,plan,output)
    shutil.copyfile(__file__,output/"report_builder.py")
    (output/"README.txt").write_text(
        "Prospective fixed-flight perception routing comparison\n\n"
        "Run from the repository root with the system Python that supplies NumPy and Matplotlib:\n"
        "PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/strong-routing-mpl /usr/bin/python3 "
        "-m experiments.strong_routing_20261030.report "
        f"--source {source} --native {native} --output {output} --supplement {Path(supplement).resolve()} --geometry {Path(geometry).resolve()}\n\n"
        "The builder refuses incomplete studies. It validates the frozen plan, native score inputs, "
        "all replay artifact hashes, and plotted success/energy/action counts before writing. "
        "report_builder.py is a source copy; explicit --source/--native/--output paths allow standalone use.\n"
        "PNG figures are 180 dpi. SVG text remains editable. Plot data, 648 primary rows and 72 separately registered supplementary rows are supplied as JSON.\n"
        "Six clips on known map assets; paired factorial replays, not independent mission trials. "
        "Native raw evidence excludes service costs. Paid outcomes are local inspection contracts. "
        "Networks, service costs and onboard energy are configured; remote service time is local time / 4.\n")
    save(output/"build_receipt.json",dict(completed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source),native=str(native),replays=len(rows),supplemental_replays=len(extra_rows),physical_clips=len(plan["cases"]),
        source_hash_count=len(hashes),plot_data_sha256=sha(output/"plot_data.json"),
        artifact_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.iterdir())
                         if p.is_file() and p.name!="build_receipt.json"}))
    print(str(output/"index.html"))
    return data


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--native",type=Path,default=DEFAULT_NATIVE)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--supplement",type=Path,default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--geometry",type=Path,default=DEFAULT_GEOMETRY)
    args=parser.parse_args()
    build(args.source,args.native,args.output,args.supplement,args.geometry)
