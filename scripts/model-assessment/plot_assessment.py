#!/usr/bin/env python3
"""Plot saved centroid metrics and render paired prediction overlays."""
from __future__ import annotations
import argparse
import csv
import json
import textwrap
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from assess_centroids import ROOT, decode_source

GROUPS = {
    'initial_model_development': [
        ('centroid_baseline','Baseline'), ('centroid_fullres_sigma5','Full res, σ5'),
        ('centroid_halfres_sigma2p5','Half res, σ2.5'), ('centroid_halfres_body','Half res + body'),
        ('centroid_fullres_body','Full res + body')],
    'cfull_parameter_tweaks': [
        ('centroid_fullres_body','Full res + body'), ('centroid_fullres_body_sigma2','Body σ2'),
        ('centroid_fullres_body_sigma3p5','Body σ3.5'), ('centroid_fullres_body_filters32','Body 32 filters'),
        ('centroid_fullres_sigma2p5','No anchor, σ2.5')],
    'synthetic_training_comparison': [
        ('centroid_fullres_body','Full res + body'), ('centroid_test_with_synthetic','2-animal synthetic'),
        ('centroid_body_synth_hard_crossings_v1','Hard crossings'),
        ('centroid_body_synth_fullframe_darkoverlap_v1','Full-frame dark overlap')],
}


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def number(row,key):
    return float(row[key]) if row[key] else float('nan')


def prism_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 14, 'axes.labelsize': 16, 'axes.labelweight': 'bold',
        'axes.titlesize': 17, 'axes.titleweight': 'bold', 'axes.titlepad': 16,
        'axes.linewidth': 2.2, 'axes.edgecolor': 'black',
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.labelsize': 13, 'ytick.labelsize': 14,
        'xtick.major.size': 7, 'ytick.major.size': 7,
        'xtick.major.width': 2.2, 'ytick.major.width': 2.2,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.pad': 7, 'ytick.major.pad': 7,
        'lines.linewidth': 2.4, 'lines.markersize': 7,
        'legend.frameon': False, 'legend.fontsize': 12,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def gradient_colors(count):
    """One sequential blue gradient, light to dark in plotted model order."""
    gradient = matplotlib.colors.LinearSegmentedColormap.from_list(
        'assessment_blue', ['#C5DCE8', '#8AB5CD', '#4B85AC', '#244E73'])
    return gradient(np.linspace(0, 1, count))


def format_percent_axis(ax, horizontal=False):
    if horizontal:
        ax.set_xlim(0,110)
        ax.set_xticks(np.arange(0,101,20))
        ax.spines['bottom'].set_bounds(0,100)
    else:
        ax.set_ylim(0,110)
        ax.set_yticks(np.arange(0,101,20))
        ax.spines['left'].set_bounds(0,100)
    ax.grid(False)


def save_plot(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


def annotate(image, rows, recovered):
    image=image.copy()
    for r in rows:
        if r['pred_x']:
            xy=(round(float(r['pred_x'])),round(float(r['pred_y'])))
            cv2.circle(image,xy,7,(255,255,0),2,cv2.LINE_AA)
        if r['gt_x']:
            xy=(round(float(r['gt_x'])),round(float(r['gt_y'])))
            color=(0,190,0) if r['status']=='TP' else (0,0,255)
            cv2.drawMarker(image,xy,color,cv2.MARKER_TILTED_CROSS,10,2,cv2.LINE_AA)
            if r['gt_index'] in recovered:
                cv2.circle(image,xy,12,(0,220,255),2,cv2.LINE_AA)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment',type=Path,default=ROOT/'outputs/model_assessment')
    parser.add_argument('--baseline',default='centroid_fullres_body')
    parser.add_argument('--candidate',default='centroid_body_synth_fullframe_darkoverlap_v1')
    parser.add_argument('--repeat',type=int,default=0,help='Zero-based repeat for qualitative images; metrics always average repeats')
    parser.add_argument('--zoom-orders',type=int,nargs='+',default=[2,3,5])
    parser.add_argument('--zoom-size',type=int,default=480)
    parser.add_argument('--plots-only',action='store_true',help='Restyle metrics without regenerating image overlays')
    args=parser.parse_args()
    prism_style()
    folder=args.assessment
    metadata=json.loads((folder/'assessment.json').read_text())
    if not 0<=args.repeat<metadata['repeats'] or args.zoom_size<1:
        parser.error('Invalid repeat or zoom size')
    out=folder/'plots';out.mkdir(exist_ok=True)
    summary=read_csv(folder/'summary.csv')
    lookup={(r['model'],r['subset']):r for r in summary}
    for name,group in GROUPS.items():
        group=[(m,label) for m,label in group if (m,'all') in lookup]
        if not group: continue
        x=np.arange(len(group))
        labels=[textwrap.fill(label,12) for _,label in group]
        colors=gradient_colors(len(group))
        title=name.replace('cfull','reference model').replace('_',' ').capitalize()
        footnote=(f"{title}. Targets: {metadata['target_protocol']}; matching ≤{metadata['tolerance_px']:g} px; "
                  f"threshold {metadata['threshold']:g}; repeats averaged. "
                  "Hard-frame recall uses the historical four-image subset.")
        (out/f'{name}_caption.txt').write_text(footnote+'\n')
        fig,axes=plt.subplots(1,2,figsize=(max(11,len(group)*2.6),4.9),sharey=True)
        for ax,subset,label in zip(axes,['all','hard'],['Overall recall','Hard-frame recall']):
            vals=[number(lookup.get((m,subset),{'recall_pct':''}),'recall_pct') for m,_ in group]
            bars=ax.bar(x,vals,.64,color=colors,edgecolor='black',linewidth=1.7)
            ax.bar_label(bars,fmt='%.1f',fontsize=13,fontweight='bold',padding=5)
            ax.set_xticks(x,labels,fontsize=12)
            format_percent_axis(ax)
            ax.set_ylabel('Recall (%)');ax.set_title(label)
            ax.tick_params(axis='y',labelleft=True)
            ax.margins(x=.09)
        fig.tight_layout(w_pad=2.6);save_plot(fig,out/name)

        fig,ax=plt.subplots(figsize=(max(5.8,len(group)*1.3),4.9))
        vals=[number(lookup[m,'all'],'precision_pct') for m,_ in group]
        bars=ax.bar(x,vals,.64,color=colors,edgecolor='black',linewidth=1.7)
        ax.bar_label(bars,fmt='%.1f',fontsize=13,fontweight='bold',padding=5)
        ax.set_xticks(x,labels,fontsize=12)
        format_percent_axis(ax)
        ax.set_ylabel('Precision (%)');ax.set_title('Precision')
        ax.margins(x=.09)
        fig.tight_layout();save_plot(fig,out/f'{name}_precision')
    # Each model appears once, in the order of the experimental decisions.
    stage_titles = {
        'initial_model_development': '1  Baseline development',
        'cfull_parameter_tweaks': '2  Parameter tweaks',
        'synthetic_training_comparison': '3  Synthetic training',
    }
    narrative_labels = {
        'centroid_fullres_body': 'Full res + body',
        'centroid_fullres_body_sigma2': 'σ = 2',
        'centroid_fullres_body_sigma3p5': 'σ = 3.5',
        'centroid_fullres_body_filters32': '32 filters',
        'centroid_fullres_sigma2p5': 'No body anchor',
        'centroid_test_with_synthetic': 'Two-animal crops',
        'centroid_body_synth_hard_crossings_v1': 'Hard-crossing scenes',
        'centroid_body_synth_fullframe_darkoverlap_v1': 'Full-frame dark overlap',
    }
    narrative,headers,seen=[],[],set()
    position=0.0
    for name,group in GROUPS.items():
        members=[(model,label) for model,label in group if (model,'all') in lookup and model not in seen]
        if not members:
            continue
        headers.append((position,stage_titles[name]))
        position+=1.0
        for model,label in members:
            narrative.append({'model':model,'label':narrative_labels.get(model,label),
                              'stage':stage_titles[name],'y':position})
            seen.add(model)
            position+=1.0
        position+=.65
    extra=[r for r in summary if r['subset']=='all' and r['model'] not in seen]
    if extra:
        headers.append((position,'4  Additional experiments'))
        position+=1.0
        for row in extra:
            narrative.append({'model':row['model'],'label':row['model'],
                              'stage':'4  Additional experiments','y':position})
            position+=1.0
    with (out/'all_models_order.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['model','label','stage','y'])
        writer.writeheader();writer.writerows(narrative)
    y=[row['y'] for row in narrative]
    colors=gradient_colors(len(narrative))

    def narrative_panel(ax,subset,key,title):
        vals=[number(lookup.get((row['model'],subset),{key:''}),key) for row in narrative]
        bars=ax.barh(y,vals,.68,color=colors,edgecolor='black',linewidth=1.5,zorder=3)
        ax.bar_label(bars,fmt='%.1f',fontsize=12,padding=5,zorder=6,
                     bbox={'facecolor':'white','edgecolor':'none','pad':1})
        ax.set_yticks(y,[row['label'] for row in narrative],fontsize=13)
        ax.tick_params(axis='y',labelleft=True)
        ax.set_ylim(position-.3,-.8)
        format_percent_axis(ax,horizontal=True)
        ax.set_xlabel('Precision (%)' if key=='precision_pct' else 'Recall (%)')
        ax.set_title(title)
        for header_y,header in headers:
            ax.text(0,header_y,header,ha='left',va='center',fontsize=13,fontweight='bold',
                    bbox={'facecolor':'white','edgecolor':'none','pad':2},zorder=5)
        ax.spines['left'].set_bounds(position-.3, y[0]-.34)
        for tick,row in zip(ax.get_yticklabels(),narrative):
            if row['model']=='centroid_fullres_body':
                tick.set_fontweight('bold')

    height=max(7,position*.5)
    fig,axes=plt.subplots(1,2,figsize=(17,height),sharey=True)
    narrative_panel(axes[0],'all','recall_pct','Overall recall')
    narrative_panel(axes[1],'hard','recall_pct','Hard-frame recall')
    fig.tight_layout(w_pad=3.0)
    save_plot(fig,out/'all_models')
    fig,ax=plt.subplots(figsize=(11,height))
    narrative_panel(ax,'all','precision_pct','Precision')
    fig.tight_layout()
    save_plot(fig,out/'all_models_precision')
    (out/'all_models_caption.txt').write_text(
        f"Experimental narrative: baseline development, independent parameter tweaks to the full-resolution body-anchor model, then synthetic training. "
        f"Each model appears once; the tweaks are not cumulative. "
        f"Targets: {metadata['target_protocol']}; matching ≤{metadata['tolerance_px']:g} px; "
        f"threshold {metadata['threshold']:g}; repeats averaged.\n")
    per_image=read_csv(folder/'per_image.csv')
    per_lookup={(r['model'],int(r['order'])):r for r in per_image}
    if (args.baseline,'all') not in lookup or (args.candidate,'all') not in lookup:
        print('Group plots saved; baseline/candidate not both present, skipping paired overlays.')
        return
    orders=sorted({int(r['order']) for r in per_image})
    fig,ax=plt.subplots(figsize=(11,4.5))
    for model,label in [(args.baseline,'Baseline'),(args.candidate,'Candidate')]:
        ax.plot(orders,[number(per_lookup[model,o],'recall_pct') for o in orders],'o-',color=gradient_colors(3)[1 if model==args.baseline else 2],label=label)
    ax.set_xticks(orders);ax.set_ylim(0,105);ax.set_xlabel('Manifest image order');ax.set_ylabel('Recall (%)')
    ax.legend(loc='lower right');fig.tight_layout();save_plot(fig,out/'paired_per_image_recall')
    sweep=read_csv(folder/'threshold_sweep.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for model in [args.baseline,args.candidate]:
        rows=[r for r in sweep if r['model']==model and r['subset']=='all']
        for ax,key in zip(axes,['recall_pct','precision_pct']):
            ax.plot([float(r['threshold']) for r in rows],[number(r,key) for r in rows],'o-',color=gradient_colors(3)[1 if model==args.baseline else 2],label='Baseline' if model==args.baseline else 'Candidate')
            ax.set_xlabel('Saved prediction score threshold');ax.set_ylabel(key.replace('_pct',' (%)'));ax.set_ylim(0,105)
    axes[0].legend(loc='lower left');fig.tight_layout(w_pad=2);save_plot(fig,out/'threshold_sweep')
    if args.plots_only:
        print(f'Wrote Prism-style PNG plots to {out}')
        return
    matches=read_csv(folder/'matches.csv')
    matches=[r for r in matches if int(r['repeat'])==args.repeat and r['model'] in [args.baseline,args.candidate]]
    overlay_dir=folder/'examples';overlay_dir.mkdir(exist_ok=True)
    zooms=[];overview=[];region_records=[]
    with h5py.File(metadata['ground_truth']) as handle:
        for order in orders:
            row=per_lookup[args.baseline,order]
            image=decode_source(handle,int(row['video_id']),int(row['source_frame']))
            first=[r for r in matches if r['model']==args.baseline and int(r['order'])==order]
            second=[r for r in matches if r['model']==args.candidate and int(r['order'])==order]
            missed={r['gt_index'] for r in first if r['status']=='FN'}
            recovered={r['gt_index'] for r in second if r['status']=='TP'} & missed
            left,right=annotate(image,first,set()),annotate(image,second,recovered)
            height,width=image.shape[:2]
            # Zoom selection is ground-truth-only: densest body-point neighbourhood.
            points=np.array([[float(r['gt_x']),float(r['gt_y'])] for r in first if r['gt_x']])
            density=(np.linalg.norm(points[:,None]-points[None,:],axis=2)<args.zoom_size/2).sum(axis=1)
            centre=points[int(np.argmax(density))]
            size=min(args.zoom_size,height,width)
            x=int(np.clip(round(centre[0]-size/2),0,width-size))
            y=int(np.clip(round(centre[1]-size/2),0,height-size))
            region_records.append({'order':order,'video_id':int(row['video_id']), 'source_frame':int(row['source_frame']),
                                   'x':x,'y':y,'width':size,'height':size})
            caption=f"Image {order}, source frame {row['source_frame']}, repeat {args.repeat} | baseline LEFT / candidate RIGHT"
            full=np.concatenate([left,right],axis=1)
            cv2.rectangle(full,(0,0),(full.shape[1],36),(255,255,255),-1)
            cv2.putText(full,caption,(8,25),0,.65,(0,0,0),1,cv2.LINE_AA)
            cv2.imwrite(str(overlay_dir/f'{order:02d}_frame_{row["source_frame"]}.png'),full)
            preview=cv2.resize(full,(1200,round(full.shape[0]*1200/full.shape[1])))
            overview.append(preview)
            if order in args.zoom_orders:
                crop=np.concatenate([left[y:y+size,x:x+size],right[y:y+size,x:x+size]],axis=1)
                crop=cv2.copyMakeBorder(crop,32,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                cv2.putText(crop,f"Image {order}, frame {row['source_frame']} | baseline / candidate",(5,22),0,.5,(0,0,0),1)
                zooms.append(crop)
    if zooms:
        cv2.imwrite(str(overlay_dir/'zoomed_comparison.png'),np.concatenate(zooms,axis=0))
    cv2.imwrite(str(overlay_dir/'all_frames_comparison.jpg'),np.concatenate(overview,axis=0))
    (overlay_dir/'regions.json').write_text(json.dumps({'baseline':args.baseline,'candidate':args.candidate,
         'repeat':args.repeat,'zoom_selection':'densest GT body-point neighbourhood, independent of predictions',
         'regions':region_records},indent=2))
    (overlay_dir/'README.txt').write_text('Baseline left; candidate right. Cyan circles: predictions. Green crosses: matched GT. Red crosses: missed GT. Yellow rings: candidate recovered a baseline miss.\nImages use one explicitly selected repeat; aggregate plots average all repeats. Zooms use the densest GT neighbourhood, not a hand-picked success.\n')
    print(f'Wrote PNG plots to {out} and comparison images to {overlay_dir}')


if __name__=='__main__':
    main()
