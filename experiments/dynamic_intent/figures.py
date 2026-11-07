"""Standalone research plots from audited summary and paid attempt logs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LABELS={'quality_current':'Quality + current intent','quality_initial':'Quality + initial intent',
 'fastest_current':'Fastest feasible','fixed_local_yolo':'Fixed local YOLO',
 'fixed_local_mask':'Fixed local Mask R-CNN','fixed_local_cabinet':'Fixed local CABiNet',
 'fixed_remote_mask':'Fixed remote Mask R-CNN'}
ORDER=list(LABELS)

def read(p):return json.loads(Path(p).read_text())

def build(root,output):
    root,output=Path(root),Path(output);summary=read(output/'summary.json')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    for metric,denom,title,file in [('mission_successes','runs','Complete three-window mission success','mission_success'),('window_successes','windows','Successful reporting windows','window_success')]:
        fig,axes=plt.subplots(1,3,figsize=(16,5),sharey=True)
        for ax,schedule in zip(axes,['no_change','urgency_increase','urgency_recovery']):
            rows=[next(r for r in summary['aggregates'] if r['scope']=='sensor_possible_clips' and r['schedule']==schedule and r['policy']==p) for p in ORDER]
            colors=['#2469a0','#d88b39']+['#829bb0']*5
            ax.barh(range(7),[r[metric] for r in rows],height=.65,color=colors)
            ax.set_yticks(range(7));ax.set_yticklabels([LABELS[p] for p in ORDER]);ax.invert_yaxis()
            limit=rows[0][denom];ax.set_xlim(0,limit*1.2);ax.set_xticks([0,limit//2,limit]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            for i,r in enumerate(rows):ax.text(r[metric]+limit*.025,i,f'{r[metric]}/{r[denom]}',va='center',fontsize=10)
            ax.set_title(schedule.replace('_',' ').title(),pad=13);ax.set_xlabel('Successful executions' if denom=='runs' else 'Successful windows')
        fig.suptitle(title,fontsize=18,y=1.03)
        fig.text(.015,-.035,'Four clips passing the optimistic sensor bound; both network and battery settings. All six clips and 504 runs remain in the report.\nCounts describe correlated replay conditions, not independent field trials.',fontsize=10,color='#506477')
        fig.tight_layout();save(fig,file)
    prefix=root/'runs/sector_revisit/sector_3/urgency_recovery/degraded/ample'
    fig,axes=plt.subplots(3,1,figsize=(13,5.8),sharex=True,gridspec_kw={'height_ratios':[1,1.2,1.2]})
    axes[0].step([0,12,24,42],[8,2,8,8],where='post',color='#18334c',lw=2.5)
    axes[0].set_ylim(0,10);axes[0].set_yticks([2,8]);axes[0].set_ylabel('Request\nbudget (s)')
    models={'CFG_MASKRCNN_1280':('#2469a0','Local Mask R-CNN'),
        'CFG_CABINET_UAVID_1280':('#338c7a','Local CABiNet'),
        'CFG_YOLO11S_960_ROT0':('#c58c27','Local YOLO'),
        'REMOTE__CFG_MASKRCNN_1280':('#835a9b','Remote Mask R-CNN'),
        'REMOTE__CFG_YOLO11S_960_ROT0':('#b57eae','Remote YOLO'),
        'REMOTE__CFG_CABINET_UAVID_1280':('#8cafb0','Remote CABiNet')}
    used=set()
    for ax,policy in zip(axes[1:],['quality_current','quality_initial']):
        events=read(prefix/policy/'events.json');row=read(prefix/policy/'row.json')
        for e in events:
            action=e['selected_action_id'];color,label=models.get(action,('#b9bec3','Abstain'));used.add(action)
            start=e['capture_time_s'];end=e['perception_completed_ns']/1e9
            ax.barh(0,end-start,left=start,height=.5,color=color,edgecolor='#d02232' if e['perception']['deadline_missed'] else color,linewidth=2)
        ax.set_yticks([]);ax.set_ylabel(LABELS[policy].replace(' + ','\n'),rotation=0,labelpad=85,va='center')
        ax.set_ylim(-.7,.7);ax.text(41,.45,f'Windows: {row["window_successes"]}/3',ha='right',fontsize=10)
    for ax in axes:
        ax.axvspan(12,24,color='#fff1d7',zorder=0)
        for t in [12,24]:ax.axvline(t,color='#999',ls='--',lw=.8)
        ax.set_xlim(0,42)
    axes[-1].set_xticks([0,6,12,18,24,30,36,42]);axes[-1].set_xlabel('Elapsed mission time (s); blanks are no active perception request')
    handles=[Patch(facecolor=models[k][0],label=models[k][1]) for k in models if k in used]
    handles.append(Patch(facecolor='white',edgecolor='#d02232',linewidth=2,label='Request missed its deadline'))
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.56,-.035),frameon=False)
    fig.suptitle('Urgency update and recovery: actual model choices',fontsize=17)
    fig.text(.22,.92,'Preselected case: sector_revisit / sector_3 | degraded link | ample battery',fontsize=10,color='#506477')
    fig.subplots_adjust(left=.23,right=.98,top=.85,bottom=.13,hspace=.3)
    save(fig,'intent_timeline')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/dynamic_intent_v1_20261031'));p.add_argument('--output',type=Path,default=Path('reports/dynamic_intent_v1_20261031'));a=p.parse_args();build(a.root,a.output)
