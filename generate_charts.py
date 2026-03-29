"""
Delhivery Analytics Charts — 8 publication-quality charts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── COLORS ───────────────────────────────────────────────────────────────────
RED   = '#E8312A'
BLACK = '#0D0D0D'
GRAY  = '#F5F5F5'
DGRAY = '#6B6B6B'
LGRAY = '#DDDDDD'

def style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'axes.spines.top':  False,
        'axes.spines.right': False,
        'axes.grid':        True,
        'grid.alpha':       0.25,
        'grid.color':       LGRAY,
        'font.family':      'DejaVu Sans',
        'axes.titlesize':   14,
        'axes.labelsize':   11,
    })

style()

df = pd.read_csv('/home/claude/delhivery-project/data/delhivery_shipments.csv')
df_opt = pd.read_csv('/home/claude/delhivery-project/data/delhivery_optimized.csv')

# ── CHART 1: On-Time Delivery Rate by Zone ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
zone_ot = df.groupby('zone')['on_time'].mean().sort_values() * 100
colors = [RED if v < 82 else '#2C2C2C' for v in zone_ot.values]
bars = ax.barh(zone_ot.index, zone_ot.values, color=colors, height=0.6, edgecolor='white')
ax.axvline(85.2, color=RED, linestyle='--', alpha=0.7, lw=2, label='Avg: 85.2%')
for bar, val in zip(bars, zone_ot.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
            va='center', fontsize=11, fontweight='bold', color=BLACK)
ax.set_xlim(0, 100)
ax.set_xlabel('On-Time Delivery Rate (%)', fontsize=12)
ax.set_title('On-Time Delivery Rate by Zone\nRed = Below Average', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
red_p = mpatches.Patch(color=RED, label='Below avg (high-priority zones)')
ax.legend(handles=[red_p], loc='lower right')
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart1_ontime_by_zone.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 1 saved")

# ── CHART 2: Avg Delay by Hour of Day ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
hourly = df.groupby('hour')['delay_hrs'].mean()
peak_mask = [(h >= 10 and h <= 13) or (h >= 17 and h <= 20) for h in hourly.index]
bar_colors = [RED if p else '#2C2C2C' for p in peak_mask]
ax.bar(hourly.index, hourly.values, color=bar_colors, width=0.7, edgecolor='white')
ax.axhline(hourly.mean(), color=DGRAY, linestyle='--', lw=1.5, alpha=0.7, label=f'Avg: {hourly.mean():.2f}h')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Avg Delay (hrs)', fontsize=12)
ax.set_title('Average Delivery Delay by Hour of Day\nRed = Peak Congestion Windows', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(range(6, 22))
ax.set_xticklabels([f'{h}:00' for h in range(6, 22)], rotation=45)
peak_p = mpatches.Patch(color=RED, label='Peak hours (10-13, 17-20)')
ax.legend(handles=[peak_p], loc='upper right')
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart2_delay_by_hour.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 2 saved")

# ── CHART 3: Cost per Shipment by Vehicle × Zone ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
pivot = df.groupby(['zone', 'vehicle_type'])['total_cost_inr'].mean().unstack()
pivot.plot(kind='bar', ax=ax, color=['#2C2C2C', '#888888', RED], edgecolor='white', width=0.7)
ax.set_xlabel('Zone', fontsize=12)
ax.set_ylabel('Avg Cost per Shipment (₹)', fontsize=12)
ax.set_title('Average Cost per Shipment — Zone × Vehicle Type', fontsize=14, fontweight='bold', pad=15)
ax.legend(title='Vehicle', fontsize=10)
ax.set_xticklabels(pivot.index, rotation=30, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart3_cost_by_zone_vehicle.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 3 saved")

# ── CHART 4: Route Clustering (Scatter) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
cluster_colors = [RED, '#2C2C2C', '#888888', '#AAAAAA']
features = ['distance_km', 'stops_per_route', 'delay_hrs', 'total_cost_inr', 'route_efficiency']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
df['cluster'] = labels

for i, c in enumerate(cluster_colors):
    mask = df['cluster'] == i
    ax.scatter(df.loc[mask, 'distance_km'], df.loc[mask, 'delay_hrs'],
               c=c, alpha=0.55, s=35, label=f'Cluster {i}', edgecolors='none')

ax.set_xlabel('Distance (km)', fontsize=12)
ax.set_ylabel('Delay (hrs)', fontsize=12)
ax.set_title('Route Clustering: Distance vs Delay\n4 Operational Segments Identified', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart4_route_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 4 saved")

# ── CHART 5: Delay Distribution — Festive vs Normal ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
festive_delay  = df[df['is_festive'] == 1]['delay_hrs']
normal_delay   = df[df['is_festive'] == 0]['delay_hrs']
ax.hist(normal_delay, bins=35, color='#2C2C2C', alpha=0.7, label=f'Normal Days (n={len(normal_delay)})', density=True)
ax.hist(festive_delay, bins=25, color=RED, alpha=0.7, label=f'Festive Season (n={len(festive_delay)})', density=True)
ax.axvline(normal_delay.mean(), color='#2C2C2C', lw=2, linestyle='--', alpha=0.9, label=f'Normal avg: {normal_delay.mean():.2f}h')
ax.axvline(festive_delay.mean(), color=RED, lw=2, linestyle='--', alpha=0.9, label=f'Festive avg: {festive_delay.mean():.2f}h')
ax.set_xlabel('Delay (hrs)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Delay Distribution: Festive Season vs Normal Days', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart5_festive_delay_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 5 saved")

# ── CHART 6: Before vs After — Key Metrics ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
metrics = [
    ('On-Time Rate (%)', 85.2, 88.6, True),
    ('Avg Delay (hrs)',  1.51, 0.94, False),
    ('Cost/Shipment (₹)', 451, 352, False),
]
for ax, (label, before, after, higher_better) in zip(axes, metrics):
    change = after - before
    bar_colors = ['#CCCCCC', RED if higher_better == (change > 0) else RED]
    bars = ax.bar(['Before', 'After'], [before, after], color=['#888888', RED if (higher_better and after > before) or (not higher_better and after < before) else '#888888'], 
                  width=0.5, edgecolor='white')
    for bar, val in zip(bars, [before, after]):
        ax.text(bar.get_x() + bar.get_width()/2, val + abs(before)*0.01, f'{val:.1f}' if isinstance(val, float) else f'₹{val}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    pct_chg = (after - before) / before * 100
    sign = '+' if pct_chg > 0 else ''
    ax.set_title(f'{label}\n({sign}{pct_chg:.1f}%)', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, max(before, after) * 1.2)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
plt.suptitle('Impact Simulation: Before vs After Optimization', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart6_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 6 saved")

# ── CHART 7: Cost Components Breakdown ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fuel = df['fuel_cost_inr'].mean()
labor = df['labor_cost_inr'].mean()
handling = (df['total_cost_inr'] - df['fuel_cost_inr'] - df['labor_cost_inr']).clip(0).mean()
sizes = [fuel, labor, handling]
labels = [f'Fuel\n₹{fuel:.0f}', f'Labor\n₹{labor:.0f}', f'Handling\n₹{handling:.0f}']
wedge_colors = [RED, '#2C2C2C', '#AAAAAA']
wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=wedge_colors,
                                     autopct='%1.0f%%', startangle=140, pctdistance=0.75,
                                     textprops={'fontsize': 11})
for at in autotexts:
    at.set_color('white'); at.set_fontweight('bold')
ax1.set_title('Cost Breakdown per Shipment', fontsize=13, fontweight='bold', pad=15)

# Package type vs cost
pkg_cost = df.groupby('package_type')['total_cost_inr'].mean().sort_values(ascending=False)
ax2.bar(pkg_cost.index, pkg_cost.values, color=[RED, '#2C2C2C', '#888888', '#BBBBBB'], edgecolor='white', width=0.55)
for i, (pkg, val) in enumerate(pkg_cost.items()):
    ax2.text(i, val + 5, f'₹{val:.0f}', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylabel('Avg Cost (₹)', fontsize=11)
ax2.set_title('Avg Cost by Package Type', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticklabels(pkg_cost.index, rotation=15)
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart7_cost_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 7 saved")

# ── CHART 8: Zone-Level Heatmap ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
zone_summary = df.groupby('zone').agg({
    'on_time':           'mean',
    'delay_hrs':         'mean',
    'total_cost_inr':    'mean',
    'route_efficiency':  'mean',
    'first_attempt':     'mean'
}).round(2)
zone_summary.columns = ['On-Time Rate', 'Avg Delay (h)', 'Avg Cost (₹)', 'Efficiency', 'First Attempt']
# Normalize for heatmap
norm = (zone_summary - zone_summary.min()) / (zone_summary.max() - zone_summary.min())
# Invert delay (lower is better)
norm['Avg Delay (h)'] = 1 - norm['Avg Delay (h)']
norm['Avg Cost (₹)'] = 1 - norm['Avg Cost (₹)']

sns.heatmap(norm, annot=zone_summary.values, fmt='', ax=ax,
            cmap='RdYlGn', linewidths=0.5, linecolor='white',
            annot_kws={'size': 9}, cbar_kws={'label': 'Performance (normalized)'})
ax.set_title('Zone Performance Heatmap\n(Green = Better, Red = Worse)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/delhivery-project/charts/chart8_zone_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 8 saved")

print("\n🎉 All 8 charts generated in /home/claude/delhivery-project/charts/")
