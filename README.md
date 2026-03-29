# 📦 Delhivery Last-Mile Delivery Optimization

> **End-to-End Analytics Project | FY 2024–25**  
> Built by: Delhivery Analytics Team

---

## 🎯 Problem Statement

Delhivery faces inefficiencies in last-mile delivery due to **suboptimal route planning**, **delivery delays**, and **high cost per shipment**.

**How can Delhivery optimize routing and delivery allocation to reduce delivery time and cost while improving on-time delivery rates?**

---

## 📊 Baseline Metrics (Before Optimization)

| Metric | Value | Target |
|--------|-------|--------|
| On-Time Delivery Rate | **85.2%** | 92% |
| Avg Delivery Delay | **1.51 hrs** | < 0.8 hrs |
| Avg Cost per Shipment | **₹451** | ₹350 |
| Cost per KM | **₹48.0** | ₹36 |
| First-Attempt Success | **88.8%** | 95%+ |
| Route Efficiency Score | **78/100** | 90+ |

---

## 📈 Impact After Optimization

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| On-Time Rate | 85.2% | **88.6%** | +3.4pp ✅ |
| Avg Delay | 1.51h | **0.94h** | −38% ✅ |
| Cost/Shipment | ₹451 | **₹352** | −22% ✅ |
| Cost per KM | ₹48.0 | **₹37.4** | −22% ✅ |
| First Attempt | 88.8% | **93.8%** | +5pp ✅ |
| Route Efficiency | 78 | **90.5** | +12.5pts ✅ |

> **💰 Estimated Annual Saving: ₹99 Crore** at 100M shipments/year scale

---

## 🗂️ Repository Structure

```
delhivery-last-mile-optimization/
│
├── 📁 data/
│   ├── delhivery_shipments.csv        # 1,200 shipment records, 26 variables
│   └── delhivery_optimized.csv        # Post-optimization simulation dataset
│
├── 📁 notebooks/
│   └── Delhivery_Optimization_Analysis.ipynb   # Full EDA + ML + Simulation
│
├── 📁 scripts/
│   ├── generate_dataset.py            # Synthetic dataset generation
│   ├── generate_charts.py             # All 8 analysis charts
│   └── build_notebook.py              # Notebook builder
│
├── 📁 dashboard/
│   └── streamlit_app.py               # Interactive ops dashboard
│
├── 📁 ppt/
│   ├── build_ppt.js                   # PowerPoint builder (PptxGenJS)
│   └── Delhivery_LastMile_Optimization.pptx   # Final 12-slide deck
│
├── 📁 charts/
│   ├── chart1_ontime_by_zone.png
│   ├── chart2_delay_by_hour.png
│   ├── chart3_cost_by_zone_vehicle.png
│   ├── chart4_route_clusters.png
│   ├── chart5_festive_delay_dist.png
│   ├── chart6_before_after.png
│   ├── chart7_cost_breakdown.png
│   └── chart8_zone_heatmap.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset + Charts
```bash
python scripts/generate_dataset.py
python scripts/generate_charts.py
```

### 3. Run the Dashboard
```bash
cd dashboard
streamlit run streamlit_app.py
```

### 4. Open the Notebook
```bash
jupyter notebook notebooks/Delhivery_Optimization_Analysis.ipynb
```

---

## 🔬 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `shipment_id` | String | Unique ID (DEL100000–DEL101199) |
| `zone` | Categorical | 8 metro zones (Delhi NCR, Mumbai…) |
| `distance_km` | Float | Last-mile delivery distance |
| `weight_kg` | Float | Package weight |
| `package_type` | Categorical | Standard / Express / Heavy / Fragile |
| `vehicle_type` | Categorical | 2-Wheeler / 3-Wheeler / Van |
| `stops_per_route` | Integer | Number of stops on the route |
| `hour` | Integer | Dispatch hour (6–21) |
| `is_weekend` | Binary | Weekend flag |
| `is_festive` | Binary | Festive season flag (Oct/Nov/Jan) |
| `peak_hour` | Binary | 1 if 10-13h or 17-20h (engineered) |
| `delay_hrs` | Float | **Target KPI** — actual vs promised |
| `on_time` | Binary | **Target KPI** — OTD flag |
| `total_cost_inr` | Float | **Target KPI** — total shipment cost |
| `route_efficiency` | Float | 0-100 composite efficiency score |
| `cluster` | Integer | K-Means route segment (4 clusters) |

---

## 🤖 Models Used

### K-Means Clustering (Route Segmentation)
- **Features:** distance_km, stops_per_route, delay_hrs, total_cost_inr, route_efficiency
- **Optimal K:** 4 (validated via Elbow + Silhouette score)
- **Segments identified:**
  - 🔴 **High-Delay Crisis** — 2.8h avg delay, Efficiency 52
  - ⚫ **Cost-Heavy Routes** — ₹642 avg cost, needs vehicle downgrade
  - 🟢 **Optimal Routes** — 1.2h delay, 86 efficiency (model for replication)
  - 🔘 **Overloaded Routes** — 25 stops/route, cap needed at 18

---

## 💡 Top 5 Product Solutions

| # | Solution | Timeline | Impact |
|---|----------|----------|--------|
| 1 | Smart Route Batching Engine | Q2 2025 | ↓ Cost ₹68/shipment, ↑ Efficiency +12pts |
| 2 | Peak-Hour Dispatch Scheduler | Q1 2025 | ↓ Delay 0.45h, ↑ OTD +2.8pp |
| 3 | Predictive Delivery Slot (ML) | Q3 2025 | ↑ 1st Attempt 88.8%→95% |
| 4 | Dynamic Vehicle Allocation | Q1 2025 | ↓ Cost/KM ₹48→₹38 |
| 5 | Zone-Level SLA Tiering | Q1 2025 | ↑ CSAT ~18% |

---

## 📊 Dashboard Features

The Streamlit dashboard includes:
- **KPI cards**: Real-time OTD%, delay, cost, efficiency
- **Zone map**: OTD rate by city with color coding
- **Hour analysis**: Delay heatmap by hour of day
- **Cost breakdown**: Fuel/labor/handling by zone & vehicle
- **Cluster scatter plot**: Interactive route segmentation
- **Before/After comparison**: Impact simulation view
- **Zone heatmap**: Multi-metric performance grid
- **Auto-insights**: NLG-style insight generation
- **Filters**: Zone, Vehicle, Package Type, Peak Hour toggle

---

## 📎 Deliverables

- ✅ **Python scripts** — dataset + analysis + charts
- ✅ **Jupyter Notebook** — full EDA + ML + simulation
- ✅ **Streamlit Dashboard** — interactive ops dashboard
- ✅ **PowerPoint (12 slides)** — consulting-grade deck with logo
- ✅ **8 publication charts** — zone analysis, clustering, before/after
- ✅ **Quantified impact** — ₹99Cr annual saving projection

---

## 📬 Contact

**Analytics Team** | Delhivery Pvt. Ltd.  
Project: Last-Mile Optimization | FY 2024–25
