import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid Windows display issues
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import shap
import hopsworks
import warnings

# Suppress version warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

# --- PROFESSIONAL THEME ---
COLORS = {
    "bg": "#0f1923",
    "card": "#1a2733",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "accent": "#38bdf8",
    "accent2": "#818cf8",
    "green": "#34d399",
    "orange": "#fb923c",
    "red": "#f87171",
    "gradient_low": "#1e3a5f",
    "gradient_high": "#ef4444"
}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["card"],
        "axes.edgecolor": COLORS["muted"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "axes.grid": True,
        "grid.color": "#2d3a4a",
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def safe_close_all():
    """Safely close all matplotlib figures"""
    try:
        plt.close('all')
    except:
        pass

def perform_enhanced_shap_analysis():
    setup_style()

    print("🔐 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv('MY_HOPSWORK_KEY'))
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("📊 Loading data from feature group (version 5)...")
    fg = fs.get_feature_group(name="karachi_aqi", version=5)
    df = fg.read()
    X = df.drop(columns=['aqi'])
    y = df['aqi']

    print("🤖 Downloading latest model...")
    # Get the latest model
    models = mr.get_models("karachi_aqi_model")
    models_sorted = sorted(models, key=lambda x: int(x.version), reverse=True)
    model_meta = models_sorted[0]
    
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    print("⚙️ Preparing data and creating SHAP explainer...")
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    # For ensemble models, extract one estimator for SHAP
    if hasattr(model, 'estimators_'):
        # model.estimators_ is a list of (name, estimator) tuples
        # Get the XGBoost model (index 1: RandomForest, XGBoost, SVR)
        xgb_tuple = model.estimators_[1]
        if isinstance(xgb_tuple, tuple):
            actual_model = xgb_tuple[1]  # Extract the XGBRegressor
        else:
            actual_model = xgb_tuple
        print(f"   Using {type(actual_model).__name__} from ensemble for SHAP analysis")
        explainer = shap.TreeExplainer(actual_model)
    else:
        print(f"   Using {type(model).__name__} for SHAP analysis")
        explainer = shap.TreeExplainer(model)

    sample_size = min(300, len(X_scaled))
    sample_X = X_scaled.head(sample_size)
    
    print(f"🔮 Computing SHAP values for {sample_size} samples...")
    shap_values = explainer.shap_values(sample_X)

    if isinstance(shap_values, list):
        sv = shap_values[1]
        ev = float(explainer.expected_value[1])
    else:
        sv = shap_values
        ev = float(np.mean(explainer.expected_value) if hasattr(explainer.expected_value, '__len__') else explainer.expected_value)

    mean_abs_shap = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)
    feature_names = X.columns.tolist()

    # Create output directory for plots
    output_dir = "shap_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # PLOT 1: EXECUTIVE SUMMARY DASHBOARD
    # =========================================================================
    print("\n📈 [1/7] Generating Executive Summary Dashboard...")
    safe_close_all()
    
    fig = plt.figure(figsize=(16, 7.5), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2], wspace=0.3,
                          left=0.04, right=0.96, top=0.85, bottom=0.12)

    ax_kpi1 = fig.add_subplot(gs[0])
    ax_kpi1.set_facecolor(COLORS["card"])
    ax_kpi1.axis('off')
    ax_kpi1.text(0.5, 0.82, "BASELINE AQI", ha='center', fontsize=10, color=COLORS["muted"], transform=ax_kpi1.transAxes)
    ax_kpi1.text(0.5, 0.48, f"{ev:.1f}", ha='center', fontsize=44, fontweight='bold', color=COLORS["accent"], transform=ax_kpi1.transAxes)
    ax_kpi1.text(0.5, 0.18, "Model's starting point\nbefore features adjust it", ha='center', fontsize=8, color=COLORS["muted"], transform=ax_kpi1.transAxes)
    for spine in ax_kpi1.spines.values():
        spine.set_color(COLORS["accent"]); spine.set_linewidth(1.5)

    ax_kpi2 = fig.add_subplot(gs[1])
    ax_kpi2.set_facecolor(COLORS["card"])
    ax_kpi2.axis('off')
    top_feat = feature_names[sorted_idx[-1]]
    top_val = mean_abs_shap[sorted_idx[-1]]
    ax_kpi2.text(0.5, 0.82, "TOP DRIVER", ha='center', fontsize=10, color=COLORS["muted"], transform=ax_kpi2.transAxes)
    ax_kpi2.text(0.5, 0.48, top_feat.upper(), ha='center', fontsize=24, fontweight='bold', color=COLORS["orange"], transform=ax_kpi2.transAxes)
    ax_kpi2.text(0.5, 0.18, f"Avg impact: +/-{top_val:.2f} AQI\non every prediction", ha='center', fontsize=8, color=COLORS["muted"], transform=ax_kpi2.transAxes)
    for spine in ax_kpi2.spines.values():
        spine.set_color(COLORS["orange"]); spine.set_linewidth(1.5)

    ax_bar = fig.add_subplot(gs[2])
    top_n = 8
    top_indices = sorted_idx[-top_n:]
    top_names = [feature_names[i] for i in top_indices]
    top_values = [mean_abs_shap[i] for i in top_indices]

    cmap = LinearSegmentedColormap.from_list("imp", [COLORS["gradient_low"], COLORS["accent"], COLORS["orange"]])
    nv = np.array(top_values)
    nv = (nv - nv.min()) / (nv.max() - nv.min() + 1e-9)
    bar_colors = [cmap(v) for v in nv]

    bars = ax_bar.barh(top_names, top_values, color=bar_colors, height=0.5, edgecolor=COLORS["card"], linewidth=1.2)
    for bar, val in zip(bars, top_values):
        ax_bar.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                   f"+/-{val:.3f}", va='center', fontsize=8, color=COLORS["text"], fontweight='bold')
    ax_bar.set_xlabel("Mean |SHAP Value|", color=COLORS["muted"], fontsize=9)
    ax_bar.set_title("Feature Importance Ranking", fontsize=12, fontweight='bold', pad=10, color=COLORS["text"])
    ax_bar.set_xlim(0, max(top_values) * 1.2)
    ax_bar.tick_params(labelsize=9)

    fig.suptitle("SHAP Executive Summary - What Drives Karachi's AQI?",
                  fontsize=15, fontweight='bold', color=COLORS["text"], y=0.96)
    
    fig.text(0.5, 0.02, "This dashboard shows the AI's baseline AQI, the top influential factor, and how all features rank by impact.",
            ha='center', fontsize=9, color=COLORS["text"], style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["muted"], alpha=0.9))
    
    filepath = os.path.join(output_dir, "1_executive_summary.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=COLORS["bg"])
    print(f"   ✅ Saved: {filepath}")
    safe_close_all()

    # =========================================================================
    # PLOT 2: BEESWARM
    # =========================================================================
    print("📈 [2/7] Generating Beeswarm Impact Chart...")
    safe_close_all()
    
    fig, ax = plt.subplots(figsize=(14, 8.5), facecolor=COLORS["bg"])
    fig.subplots_adjust(left=0.18, right=0.88, top=0.92, bottom=0.12)
    
    ax.set_facecolor(COLORS["card"])
    shap.summary_plot(sv, sample_X, plot_type="beeswarm", max_display=10, show=False, cmap="RdBu_r")

    ax.set_title("Does a HIGH or LOW feature value push AQI UP or DOWN?",
                  fontsize=11, fontweight='bold', pad=25, color=COLORS["text"])
    ax.set_xlabel("SHAP Value  ->  Pushes AQI Higher", fontsize=9, color=COLORS["muted"])

    if len(fig.axes) > 1:
        cbar = fig.axes[-1]
        cbar.tick_params(colors=COLORS["text"], labelsize=8)
        cbar.set_ylabel("Feature Value", color=COLORS["text"], fontsize=9)

    props = dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["accent"], alpha=0.9)
    ax.text(0.015, 0.96, "Red = High value\nBlue = Low value\nRight = AQI Up\nLeft = AQI Down",
            transform=ax.transAxes, fontsize=8, color=COLORS["text"], va='top', bbox=props)

    fig.suptitle("SHAP Beeswarm Analysis", fontsize=15, fontweight='bold', color=COLORS["text"], y=0.96)
    
    fig.text(0.5, 0.02, "Each dot is one prediction - red means high feature value, blue means low; position shows whether it pushed AQI up or down.",
            ha='center', fontsize=9, color=COLORS["text"], style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["muted"], alpha=0.9))
    
    filepath = os.path.join(output_dir, "2_beeswarm.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=COLORS["bg"])
    print(f"   ✅ Saved: {filepath}")
    safe_close_all()

    # =========================================================================
    # PLOT 3: DEPENDENCE PLOTS
    # =========================================================================
    print("📈 [3/7] Generating Dependence Plots...")
    safe_close_all()
    
    top3_features = [feature_names[i] for i in sorted_idx[-3:]][::-1]

    feature_descriptions = {
        "pm25": "Fine particulate matter - the most dangerous pollutant",
        "pm10": "Coarse dust particles - worsens respiratory health",
        "co": "Carbon monoxide - from vehicle emissions",
        "dew_point": "Humidity traps pollutants near the ground",
        "wind_speed": "Wind disperses airborne pollutants",
        "o3": "Ground-level ozone - from sunlight reactions",
        "no2": "Nitrogen dioxide - car and industrial exhaust",
        "so2": "Sulphur dioxide - from industrial burning",
        "humidity": "Affects pollutant concentration",
        "temperature": "Influences air chemical reactions",
        "aqi_lag_1": "Previous hour's AQI - pollution persists",
        "aqi_lag_2": "AQI from 2 hours ago - persistence",
        "aqi_change_rate": "How fast AQI is changing",
        "pm25_lag_1": "Previous PM2.5 - dust persists",
        "hour": "Time of day - activity cycles",
        "weekday": "Weekday vs weekend differences",
        "year": "Year", "month": "Month", "day": "Day"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=COLORS["bg"])
    fig.subplots_adjust(left=0.05, right=0.96, top=0.82, bottom=0.15, wspace=0.32)

    for i, feat in enumerate(top3_features):
        ax = axes[i]
        ax.set_facecolor(COLORS["card"])
        feat_idx = feature_names.index(feat)

        interaction_idx = shap.utils.approximate_interactions(feat_idx, sv, sample_X)[0]
        interaction_name = feature_names[interaction_idx]

        x_vals = sample_X[feat].values
        shap_vals = sv[:, feat_idx]
        color_vals = sample_X[interaction_name].values

        scatter = ax.scatter(x_vals, shap_vals, c=color_vals, cmap="coolwarm",
                            s=25, alpha=0.7, edgecolors=COLORS["card"], linewidths=0.4,
                            vmin=np.percentile(color_vals, 5), vmax=np.percentile(color_vals, 95))

        sort_order = np.argsort(x_vals)
        window = max(1, len(x_vals) // 15)
        x_sm = pd.Series(x_vals[sort_order]).rolling(window, center=True).mean().values
        y_sm = pd.Series(shap_vals[sort_order]).rolling(window, center=True).mean().values
        ax.plot(x_sm, y_sm, color=COLORS["accent"], linewidth=2.2, label="Trend", zorder=5)

        ax.axhline(0, color=COLORS["muted"], linestyle="--", linewidth=0.7, alpha=0.5)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.02)
        cbar.set_label(interaction_name, fontsize=7, color=COLORS["muted"])
        cbar.ax.tick_params(colors=COLORS["muted"], labelsize=6)

        ax.set_xlabel(feat, fontsize=10, fontweight='bold', color=COLORS["text"])
        ax.set_ylabel("SHAP Value" if i == 0 else "", fontsize=8, color=COLORS["muted"])
        ax.set_title(f"#{i+1} {feat.upper()}\n{feature_descriptions.get(feat, '')}",
                    fontsize=9, fontweight='bold', color=COLORS["accent"], pad=6, linespacing=1.3)
        ax.legend(loc="upper left", fontsize=7, frameon=True,
                 facecolor=COLORS["card"], edgecolor=COLORS["muted"], labelcolor=COLORS["text"])
        ax.tick_params(labelsize=7)

    fig.suptitle("Feature Dependence - How Does Each Feature's Value Change Its Effect on AQI?",
                  fontsize=14, fontweight='bold', color=COLORS["text"], y=0.97)
    
    fig.text(0.5, 0.02, "The trend line shows how a feature's value relates to its AQI impact; dot colors reveal which other feature interacts with it.",
            ha='center', fontsize=9, color=COLORS["text"], style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["muted"], alpha=0.9))
    
    filepath = os.path.join(output_dir, "3_dependence_plots.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=COLORS["bg"])
    print(f"   ✅ Saved: {filepath}")
    safe_close_all()

    # =========================================================================
    # PLOT 4: WATERFALL - PEAK POLLUTION
    # =========================================================================
    print("📈 [4/7] Generating Waterfall Plot (Peak Pollution Event)...")
    safe_close_all()
    
    high_idx = y.head(sample_size).idxmax()
    explanation = shap.Explanation(values=sv[high_idx], base_values=ev, data=X.iloc[high_idx].values, feature_names=X.columns.tolist())

    fig, ax = plt.subplots(figsize=(13, 8.5), facecolor=COLORS["bg"])
    fig.subplots_adjust(left=0.28, right=0.96, top=0.85, bottom=0.12)
    
    ax.set_facecolor(COLORS["card"])
    shap.plots.waterfall(explanation, max_display=12, show=False)
    ax.tick_params(labelsize=8)

    props = dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["red"], alpha=0.9)
    ax.text(0.02, 0.96, f"Peak AQI Event\nBaseline: {ev:.1f} -> Final: {y[high_idx]:.1f}\nNet push: +{y[high_idx] - ev:.1f}",
           transform=ax.transAxes, fontsize=8, color=COLORS["text"], va='top', bbox=props)

    fig.suptitle("SHAP Waterfall - Peak Pollution Event Analysis",
                  fontsize=15, fontweight='bold', color=COLORS["text"], y=0.97)
    
    fig.text(0.5, 0.02, "Starting from baseline AQI, each bar shows how much one factor pushed the prediction higher (red) or lower (blue) to reach the final value.",
            ha='center', fontsize=9, color=COLORS["text"], style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["muted"], alpha=0.9))
    
    filepath = os.path.join(output_dir, "4_waterfall_peak.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=COLORS["bg"])
    print(f"   ✅ Saved: {filepath}")
    safe_close_all()

    # =========================================================================
    # PLOT 5 & 6: CLEAN vs POLLUTED DAYS
    # =========================================================================
    print("📈 [5-6/7] Generating Clean vs Polluted Day Comparison...")
    low_idx = y.head(sample_size).idxmin()

    scenarios = [
        (low_idx, f"Cleanest Day (AQI: {y[low_idx]:.1f})", COLORS["green"], "clean"),
        (high_idx, f"Worst Pollution Day (AQI: {y[high_idx]:.1f})", COLORS["red"], "polluted")
    ]
    captions = [
        "On this clean day, low PM2.5 and favorable wind conditions kept pollutants dispersed - these factors pulled AQI down from baseline.",
        "On this polluted day, high PM2.5 and trapped humidity pushed AQI well above baseline - these are the conditions residents should watch for."
    ]

    for i, (idx, title, border_color, scenario_name) in enumerate(scenarios):
        safe_close_all()
        
        exp = shap.Explanation(values=sv[idx], base_values=ev, data=X.iloc[idx].values, feature_names=X.columns.tolist())

        fig, ax = plt.subplots(figsize=(13, 8.5), facecolor=COLORS["bg"])
        fig.subplots_adjust(left=0.28, right=0.96, top=0.82, bottom=0.12)
        
        ax.set_facecolor(COLORS["card"])
        for spine in ax.spines.values():
            spine.set_color(border_color); spine.set_linewidth(2)

        shap.plots.waterfall(exp, max_display=8, show=False)
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS["text"], pad=10)
        ax.tick_params(labelsize=8)

        fig.suptitle("Side-by-Side Comparison: Clean vs Polluted Day",
                      fontsize=15, fontweight='bold', color=COLORS["text"], y=0.97)
        
        fig.text(0.5, 0.02, captions[i], ha='center', fontsize=9, color=COLORS["text"], style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["muted"], alpha=0.9))
        
        filepath = os.path.join(output_dir, f"5_waterfall_{scenario_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=COLORS["bg"])
        print(f"   ✅ Saved: {filepath}")
        safe_close_all()

    # =========================================================================
    # PLOT 7: BAR CHART - ALL FEATURES RANKED
    # =========================================================================
    print("📈 [7/7] Generating Complete Feature Importance Bar Chart...")
    safe_close_all()
    
    # Sort all features by importance
    all_sorted_idx = sorted_idx
    all_names = [feature_names[i] for i in all_sorted_idx]
    all_values = [mean_abs_shap[i] for i in all_sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(all_names) * 0.35)), facecolor=COLORS["bg"])
    fig.subplots_adjust(left=0.2, right=0.95, top=0.93, bottom=0.08)
    
    ax.set_facecolor(COLORS["card"])
    
    # Create gradient colors
    cmap = LinearSegmentedColormap.from_list("imp", [COLORS["gradient_low"], COLORS["accent"], COLORS["orange"]])
    nv = np.array(all_values)
    nv = (nv - nv.min()) / (nv.max() - nv.min() + 1e-9)
    bar_colors = [cmap(v) for v in nv]
    
    bars = ax.barh(all_names, all_values, color=bar_colors, height=0.6, edgecolor=COLORS["card"], linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, all_values):
        ax.text(bar.get_width() + max(all_values) * 0.01, bar.get_y() + bar.get_height()/2,
               f"{val:.4f}", va='center', fontsize=7, color=COLORS["text"], fontweight='bold')
    
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on AQI)", color=COLORS["text"], fontsize=10, fontweight='bold')
    ax.set_ylabel("Features", color=COLORS["text"], fontsize=10, fontweight='bold')
    ax.set_xlim(0, max(all_values) * 1.15)
    ax.tick_params(labelsize=8, colors=COLORS["text"])
    ax.grid(axis='x', alpha=0.3, color=COLORS["muted"])
    
    fig.suptitle("Complete Feature Importance Ranking - All Features by SHAP Impact",
                  fontsize=14, fontweight='bold', color=COLORS["text"], y=0.98)
    
    fig.text(0.5, 0.02, "Higher values mean the feature has a stronger average influence on AQI predictions (either pushing up or down).",
            ha='center', fontsize=9, color=COLORS["text"], style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["card"], edgecolor=COLORS["muted"], alpha=0.9))
    
    filepath = os.path.join(output_dir, "7_all_features_ranked.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=COLORS["bg"])
    print(f"   ✅ Saved: {filepath}")
    safe_close_all()

    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("\n" + "="*80)
    print("✨ ALL SHAP VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\n📁 Output Directory: {os.path.abspath(output_dir)}/")
    print("\n📊 Generated Files:")
    print("   1. 1_executive_summary.png      - KPI dashboard with baseline & top features")
    print("   2. 2_beeswarm.png               - Feature value distribution vs SHAP impact")
    print("   3. 3_dependence_plots.png       - Top 3 features interaction analysis")
    print("   4. 4_waterfall_peak.png         - Peak pollution event breakdown")
    print("   5. 5_waterfall_clean.png        - Clean day analysis")
    print("   6. 5_waterfall_polluted.png     - Polluted day analysis")
    print("   7. 7_all_features_ranked.png    - Complete feature importance ranking")
    print("\n🎨 All plots saved at 300 DPI for publication quality!")
    print("="*80)

if __name__ == "__main__":
    perform_enhanced_shap_analysis()
