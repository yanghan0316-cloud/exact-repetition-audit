"""Publication figure from the completed n=4 experiment; no estimated data."""
import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/ngram_blocking/analysis.json'
OUT = ROOT / 'local/ngram_figures'
FIG_DATA = ROOT / 'local/ngram_figures/data'


def main():
    global DATA, OUT, FIG_DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--data-dir', type=Path, default=FIG_DATA)
    args = parser.parse_args()
    DATA, OUT, FIG_DATA = args.analysis, args.output_dir, args.data_dir
    result = json.loads(DATA.read_text(encoding='utf-8'))
    assert result['status'] == 'complete'
    arms = result['arms']
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['Times New Roman'], 'font.size':8.5,
        'axes.labelsize':8.5, 'axes.titlesize':10, 'xtick.labelsize':8,
        'ytick.labelsize':8.5, 'legend.fontsize':8, 'pdf.fonttype':42,
        'ps.fonttype':42, 'axes.linewidth':0.65, 'savefig.dpi':320,
        'axes.unicode_minus':False,
    })
    fig = plt.figure(figsize=(7.15, 2.63))
    ax1 = fig.add_axes([0.085, 0.27, 0.37, 0.53])
    ax2 = fig.add_axes([0.59, 0.27, 0.295, 0.53])
    palette = {'substitutions':'#0072B2', 'deletions':'#E69F00', 'insertions':'#D55E00'}
    labels = {'substitutions':'Substitutions', 'deletions':'Deletions', 'insertions':'Insertions'}
    changes = []
    for y, (arm, comparison) in zip([1, 0], [('B','B - A'), ('C','C - A')]):
        positive = negative = 0.0
        row = {'comparison':comparison}
        for key, color in palette.items():
            count = arms[arm][key] - arms['A'][key]
            amount = 100 * count / result['reference_characters']
            row[key + '_count_delta'] = count
            row[key + '_percentage_points'] = amount
            start = positive if amount >= 0 else negative
            ax1.barh(y, amount, left=start, height=0.30, color=color,
                     edgecolor='white', linewidth=0.35)
            if amount >= 0:
                positive += amount
            else:
                negative += amount
        net = 100 * (arms[arm]['micro_cpcer'] - arms['A']['micro_cpcer'])
        ax1.plot(net, y, marker='D', color='#202020', markersize=4.3, zorder=5)
        ax1.text(net, y + 0.26, f'{net:+.2f}', ha='center', va='bottom', fontsize=8.5)
        row['net_percentage_points'] = net
        changes.append(row)
    ax1.set_yticks([1,0], ['B - A\nGuard','C - A\nBlocking'])
    ax1.set_ylim(-0.52,1.65)
    ax1.set_xlim(-2,6)
    ax1.set_xticks([-2,0,2,4,6])
    ax1.set_xlabel('Change in error rate (percentage points)', labelpad=4)
    ax1.axvline(0, color='#717171', lw=0.65, zorder=0)
    ax1.xaxis.grid(True, color='#DDE1E5', linewidth=0.45, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='y', length=0, pad=5)
    ax1.tick_params(axis='x', length=3)

    contrasts = []
    for y, label in zip([2,1,0], ['C-A','D-C','C-B']):
        row = result['comparisons'][label]
        mean = row['delta_percentage_points']
        low, high = row['ci95_lower_percentage_points'], row['ci95_upper_percentage_points']
        color = '#A34A20' if mean > 0 else '#404040'
        ax2.errorbar(mean,y,xerr=np.array([[mean-low],[high-mean]]),fmt='o',
                     color=color,markersize=4.0,capsize=2.8,elinewidth=1.05,capthick=0.85)
        ax2.text(mean,y+0.26,f'{mean:+.2f}',ha='center',va='bottom',fontsize=8.3)
        ax2.text(1.045,y,f"{row['wins']}/{row['ties']}/{row['losses']}",
                 transform=ax2.get_yaxis_transform(),ha='left',va='center',fontsize=8.1)
        contrasts.append({'comparison':label, **{k:row[k] for k in (
            'delta_percentage_points','ci95_lower_percentage_points',
            'ci95_upper_percentage_points','wins','ties','losses')}})
    ax2.text(1.045,2.58,'W/T/L',transform=ax2.get_yaxis_transform(),ha='left',va='bottom',fontsize=8)
    ax2.set_yticks([2,1,0],['C - A','D - C','C - B'])
    ax2.set_ylim(-0.60,3.05)
    ax2.set_xlim(-1,8.5)
    ax2.set_xticks([0,2,4,6,8])
    ax2.set_xlabel('Micro-cpCER difference (percentage points)',labelpad=4)
    ax2.axvline(0,color='#717171',lw=0.75,linestyle=(0,(3,2)))
    ax2.xaxis.grid(True,color='#DDE1E5',linewidth=0.45)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='y',length=0,pad=5)
    ax2.tick_params(axis='x',length=3)
    for ax in (ax1, ax2):
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#717171')
    fig.text(0.085,0.91,'(a) Where the errors change',weight='bold',fontsize=10)
    fig.text(0.59,0.91,'(b) Paired effects and uncertainty',weight='bold',fontsize=10)
    legend = [Patch(facecolor=color,label=labels[key]) for key,color in palette.items()]
    legend += [Line2D([0],[0],marker='D',linestyle='none',color='#202020',markersize=4,label='Net change')]
    fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(0.075,0.006),ncol=4,
               frameon=False,handlelength=1.0,columnspacing=1.15,handletextpad=0.45)
    fig.text(0.59,0.065,'Points: paired differences   Bars: 95% bootstrap intervals',fontsize=7.6)
    OUT.mkdir(parents=True,exist_ok=True)
    FIG_DATA.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT / f'ngram_control_tradeoff.{ext}',facecolor='white')
    plt.close(fig)
    for name, rows in [('error_composition',changes),('paired_effects',contrasts)]:
        with (FIG_DATA / f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (FIG_DATA/'source.json').write_text(json.dumps({'source':str(DATA),'reference_characters':result['reference_characters'],
        'scope':result['scope'],'bootstrap':result['bootstrap'],'error_composition':changes,'paired_effects':contrasts},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figure':str(OUT/'ngram_control_tradeoff.pdf'),'reference_chars':result['reference_characters']}))


if __name__ == '__main__':
    main()
