# %%
import os
import time
import platform
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# %%
# 1. DIRECTORY SETUP
OUTPUT_DIR = "Calibrated_outputs"
OPTIMIZATION_DIR = os.path.join(OUTPUT_DIR, "optimization")
MODEL_DIR = os.path.join(OUTPUT_DIR, "optimal_model")
FIGURES_DIR = os.path.join(MODEL_DIR, "figures")
EXCEL_DIR = os.path.join(MODEL_DIR, "excel_data")

for d in [OPTIMIZATION_DIR, MODEL_DIR, FIGURES_DIR, EXCEL_DIR]: os.makedirs(d, exist_ok=True)

def long_path(path):
    """
    Bypass Windows' 260-character MAX_PATH limit by prefixing the absolute
    path with \\\\?\\ (the extended-length path marker). No admin rights or
    registry change needed. No-op on non-Windows systems.
    """
    if platform.system() == "Windows":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            abs_path = "\\\\?\\" + abs_path
        return abs_path
    return path

def safe_to_excel(df, path, index=False, retries=4, delay=0.5):
    """
    Write an Excel file, recreating the parent folder and retrying if it's
    briefly missing/locked (OneDrive sync, antivirus scanning, etc.), and
    using long_path() so a near-260-character full path doesn't silently
    fail on Windows.
    """
    folder = os.path.dirname(path)
    for attempt in range(retries):
        try:
            os.makedirs(folder, exist_ok=True)
            df.to_excel(long_path(path), index=index)
            return
        except FileNotFoundError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)

def safe_savefig(fig, path, dpi=300):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fig.savefig(long_path(path), dpi=dpi)

# 2. DATA LOADING and FEATURE VERIFICATION

DATA_PATH =  "Multiporttube_V2.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="MLinput")

TARGET = "dPdz_fr_exp" if "dPdz_fr_exp" in df.columns else "dpdzfr_Exp"

if 'BoPh_Pf' not in df.columns:
    if 'Bo' in df.columns and 'Ph_Pf' in df.columns:
        df['BoPh_Pf'] = df['Bo'] * df['Ph_Pf']


SELECTED_FEATURES = ['Frtp', 'Beta', 'Prtp', 'Refo', 'BoPh_Pf', 'Wefo', 'Sugo']

missing_feats = [f for f in SELECTED_FEATURES if f not in df.columns]
if missing_feats:
    raise KeyError(f"Missing features from ML sheet: {missing_feats}")

X = df[SELECTED_FEATURES].reset_index(drop=True)
y_raw = df[TARGET].reset_index(drop=True)
y = np.log(y_raw)  

fluids = df["fluid"].reset_index(drop=True) if "fluid" in df.columns else df["Fluid"].reset_index(drop=True)
studies = df["Study"].reset_index(drop=True) if "Study" in df.columns else fluids

# (< 55 samples get 3.0, others 1.0)
fluid_counts = fluids.value_counts()
sample_weights = fluids.map(lambda f: 3.0 if fluid_counts[f] < 55 else 1.0).values

print(f"Loaded dataset: {DATA_PATH} | Total samples: {len(X)}")
print(f"Explicit Features Used: {SELECTED_FEATURES}")


# 3. 80/20 UNTOUCHED HOLDOUT SPLIT & FINAL MODEL TRAINING

print("\n--- Establishing 80/20 Untouched Holdout Split & Training Model ---")
X_train_val, X_test, y_train_val, y_test, w_train_val, w_test, f_train_val, f_test = train_test_split(
    X, y, sample_weights, fluids, test_size=0.20, random_state=42, stratify=fluids)

X_train, X_val, y_train, y_val, w_train, w_val, f_train, f_val = train_test_split(
    X_train_val, y_train_val, w_train_val, f_train_val, test_size=0.10, random_state=42, stratify=f_train_val)

#Model Parameters
RF_ESTIMATORS = 15
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 5
LR = 0.01
N_ESTIMATORS = 400

base_learner = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=MAX_DEPTH, 
                                     min_samples_leaf=MIN_SAMPLES_LEAF, random_state=42, n_jobs=None)

final_model = NGBRegressor(Dist=Normal, Score=LogScore, Base=base_learner,
                           n_estimators=N_ESTIMATORS, learning_rate=LR, random_state=42, verbose=False)
#do the predictions on train data
final_model.fit(X_train, y_train, sample_weight=w_train)


# 4. EVALUATE the model predictions with ERROR BARS, & RELIABILITY DIAGRAMS

def evaluate_and_save_split(X_split, y_split, split_name, fluid_labels, s=1.0):
    """
    s: post-hoc variance-scaling ("temperature scaling") factor for
       probabilistic regression (Kuleshov, Fenner & Ermon 2018, ICML;
       Levi, Gispan, Giladi & Fetaya 2022, Sensors 22(15):5540).
       s=1.0 reproduces the ORIGINAL, uncalibrated behavior exactly, so
       every call below with the default argument is byte-for-byte identical
       to the pre-calibration pipeline. Point predictions (pred_log/pred_orig),
       MAPE, and R2 are NEVER touched by s -- only the predictive std used
       for the CI bounds and the reliability diagram.
    """
    os.makedirs(EXCEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pred_log = final_model.predict(X_split)
    dist = final_model.pred_dist(X_split)
    
    pred_clipped = np.clip(pred_log, np.log(y_raw.min() * 0.1), np.log(y_raw.max() * 10.0))
    pred_orig = np.exp(pred_clipped)
    true_orig = np.exp(y_split)
    
    # Calculate exact physical upper and lower bounds using e^x
    log_std = dist.scale
    calibrated_std = s * log_std
    lower_bound = np.exp(pred_log - (1.96 * calibrated_std))
    upper_bound = np.exp(pred_log + (1.96 * calibrated_std))
    
    tag = split_name if s == 1.0 else f"{split_name}_calibrated"
    title_suffix = f"{split_name.capitalize()}" + (f" (Calibrated, s={s:.3f})" if s != 1.0 else "")
    
    report_df = pd.DataFrame({"Fluid": fluid_labels.values, "True_Value": true_orig.values,
        "Predicted_Value": pred_orig, "Log_Standard_Deviation": log_std,
        "Calibration_Factor_s": s, "Calibrated_Log_Std": calibrated_std,
        "Lower_Bound_95CI": lower_bound, "Upper_Bound_95CI": upper_bound })
    
    safe_to_excel(report_df, os.path.join(EXCEL_DIR, f"final_opt_{tag}_uncertainty.xlsx"))
    
    # Error Bar Plot (True asymmetric bounds)
    yerr_lower = pred_orig - lower_bound
    yerr_upper = upper_bound - pred_orig
    
    #plots
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(true_orig, pred_orig, yerr=[yerr_lower, yerr_upper], fmt='o', color='teal', ecolor='darkcyan',
                elinewidth=0.8, alpha=0.5, label="Prediction 95% CI")
    min_val, max_val = min(true_orig.min(), pred_orig.min()), max(true_orig.max(), pred_orig.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal Fit (1:1)")
    ax.set_xlabel("Measured Pressure Drop", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Pressure Drop", fontsize=11, fontweight="bold")
    ax.set_title(f"Final Model ({title_suffix}): Error Bars (95% CI)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIGURES_DIR, f"final_opt_{tag}_error_bars.png"))
    plt.close(fig)
    
    # Reliability Diagram
    confidence_levels = np.linspace(0.1, 0.9, 9)
    empirical_coverages = []
    
    for level in confidence_levels:
        z = norm.ppf(0.5 + level / 2.0)
        lower = pred_log - z * calibrated_std 
        upper = pred_log + z * calibrated_std 
        in_bounds = (y_split >= lower) & (y_split <= upper)
        empirical_coverages.append(np.mean(in_bounds))
        
    calib_df = pd.DataFrame({
        "Nominal_Confidence": confidence_levels,
        "Empirical_Coverage": empirical_coverages})
    safe_to_excel(calib_df, os.path.join(EXCEL_DIR, f"final_opt_{tag}_calibration_data.xlsx"))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Ideal Calibration (45° Line)")
    ax.plot(confidence_levels, empirical_coverages, 'bo-', lw=2, markersize=6, label=f"Model {title_suffix}")
    ax.set_xlabel("Nominal Confidence Level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Coverage", fontsize=11, fontweight="bold")
    ax.set_title(f"Reliability Diagram / Calibration ({title_suffix})", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIGURES_DIR, f"final_opt_{tag}_reliability.png"))
    plt.close(fig)
    
    mape = mean_absolute_percentage_error(true_orig, pred_orig) * 100
    r2 = r2_score(true_orig, pred_orig)
    print(f"[{tag.upper()}] MAPE: {mape:.2f}% | R²: {r2:.4f} | s={s:.3f}")

evaluate_and_save_split(X_train, y_train, "train", f_train)
evaluate_and_save_split(X_val, y_val, "validation", f_val)
evaluate_and_save_split(X_test, y_test, "test_untouched", f_test)


# 5. LEAVE-ONE-FLUID-OUT (LOFO) 

def run_group_evaluation(group_labels, group_name_str):
    os.makedirs(EXCEL_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logo = LeaveOneGroupOut()
    results = []
    all_group_preds = []
    
    print(f"\n--- Running Leave-One-{group_name_str}-Out Evaluation ---")
    for train_idx, test_idx in logo.split(X, y, groups=group_labels):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        w_tr = sample_weights[train_idx]
        group_val = group_labels.iloc[test_idx].iloc[0]
        
        base_learner = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=MAX_DEPTH, 
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=42, n_jobs=None)
        # FIXED: Dist=Normal here as well
        model = NGBRegressor(Dist=Normal, Score=LogScore, Base=base_learner, n_estimators=N_ESTIMATORS,
                             learning_rate=LR, random_state=42, verbose=False)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        pred_log = model.predict(X_te)
        dist = model.pred_dist(X_te)
        
        pred_clipped = np.clip(pred_log, np.log(y_raw.min() * 0.1), np.log(y_raw.max() * 10.0))
        pred_orig = np.exp(pred_clipped)
        true_orig = np.exp(y_te)
        
        # FIXED: Exact physical upper and lower bounds
        log_std = dist.scale
        lower_bound = np.exp(pred_log - (1.96 * log_std))
        upper_bound = np.exp(pred_log + (1.96 * log_std))
       
        mape = mean_absolute_percentage_error(true_orig, pred_orig) * 100
        r2 = r2_score(true_orig, pred_orig) if len(true_orig) > 1 else np.nan
        
        results.append({f"Held_Out_{group_name_str}": group_val, "Samples": len(y_te), "MAPE_%": mape, "R2": r2})
        
        group_df = pd.DataFrame({ f"Held_Out_{group_name_str}": [group_val] * len(true_orig),
            "True_Value": true_orig.values,"Predicted_Value": pred_orig,
            "Log_Standard_Deviation": log_std, "Lower_Bound_95CI": lower_bound,
            "Upper_Bound_95CI": upper_bound, "APE_%": np.abs(pred_orig - true_orig.values) / true_orig.values * 100})
        all_group_preds.append(group_df)
        print(f"Held-out {group_name_str}: {str(group_val):<15} | Samples: {len(y_te):3d} | MAPE: {mape:6.2f}%")
        
    res_df = pd.DataFrame(results).sort_values(by="MAPE_%", ascending=False)
    safe_to_excel(res_df, os.path.join(EXCEL_DIR, f"final_lo{group_name_str[0]}o_{group_name_str.lower()}_summary.xlsx"))
    
    combined_df = pd.concat(all_group_preds, ignore_index=True)
    safe_to_excel(combined_df, os.path.join(EXCEL_DIR, f"final_lo{group_name_str[0]}o_{group_name_str.lower()}_all_predictions_and_errors.xlsx"))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['crimson' if m > 40 else 'royalblue' for m in res_df["MAPE_%"]]
    ax.bar(range(len(res_df)), res_df["MAPE_%"], color=colors, edgecolor="black")
    ax.axhline(40, color="darkred", linestyle="--", linewidth=1.5, label="40% Error Threshold")
    ax.set_xticks(range(len(res_df)))
    ax.set_xticklabels(res_df[f"Held_Out_{group_name_str}"], rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("MAPE (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Leave-One-{group_name_str}-Out Generalization MAPE", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    safe_savefig(fig, os.path.join(FIGURES_DIR, f"final_lo{group_name_str[0]}o_mape_by_{group_name_str.lower()}.png"))
    plt.close(fig)

    return res_df, combined_df

lofo_summary_df, lofo_preds_df = run_group_evaluation(fluids, "Fluid")
loso_summary_df, loso_preds_df = run_group_evaluation(studies, "Study")

print(f"\n--- Final Corrected Pipeline Completed Successfully. Outputs saved to ./{OUTPUT_DIR}/ ---")

# %%
# UNCERTAINTY CALIBRATION 

from scipy.optimize import minimize_scalar

def fit_variance_scale(model, X_cal, y_cal, s_bounds=(1.0, 6.0)):
    """
    Fit scalar s >= 1 minimizing Gaussian NLL of the model's predictive
    distribution on a calibration split (here: validation).
    Returns (s_optimal, nll_at_s1, nll_at_s_optimal).
    """
    pred_log_cal = model.predict(X_cal)
    log_std_cal = model.pred_dist(X_cal).scale
    y_true_cal = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)

    def neg_log_likelihood(s):
        scaled_std = s * log_std_cal
        return np.mean(0.5 * np.log(2 * np.pi * scaled_std ** 2)
            + 0.5 * ((y_true_cal - pred_log_cal) ** 2) / (scaled_std ** 2) )

    res = minimize_scalar(neg_log_likelihood, bounds=s_bounds, method="bounded")
    return res.x, neg_log_likelihood(1.0), res.fun

CALIBRATION_S, val_nll_raw, val_nll_calibrated = fit_variance_scale(final_model, X_val, y_val)

print("--- Post-Hoc Uncertainty Calibration (fit on VALIDATION only) ---")
print(f"Calibration factor s               : {CALIBRATION_S:.4f}")
print(f"Validation NLL, uncalibrated (s=1) : {val_nll_raw:.4f}")
print(f"Validation NLL, calibrated         : {val_nll_calibrated:.4f}")
if CALIBRATION_S >= 5.999:
    print("WARNING: s pinned at the upper search bound -- widen s_bounds and refit; "
          "the true optimum may be larger.")

# %%
print(f"\n--- Re-generating calibrated diagnostics (s = {CALIBRATION_S:.4f}) ---")
evaluate_and_save_split(X_train, y_train, "train", f_train, s=CALIBRATION_S)
evaluate_and_save_split(X_val, y_val, "validation", f_val, s=CALIBRATION_S)
evaluate_and_save_split(X_test, y_test, "test_untouched", f_test, s=CALIBRATION_S)

# %%
def check_ood_coverage_transfer(preds_df, group_col, s=CALIBRATION_S, level=0.95):
    z = norm.ppf(0.5 + level / 2.0)
    pred_log = np.log(preds_df["Predicted_Value"].values)
    true_log = np.log(preds_df["True_Value"].values)
    raw_std = preds_df["Log_Standard_Deviation"].values
    scaled_std = s * raw_std

    rows = []
    for g in preds_df[group_col].unique():
        mask = preds_df[group_col].values == g
        cov_raw = np.mean(np.abs(true_log[mask] - pred_log[mask]) <= z * raw_std[mask])
        cov_scaled = np.mean(np.abs(true_log[mask] - pred_log[mask]) <= z * scaled_std[mask])
        rows.append({group_col: g, "N": int(mask.sum()),
                     f"Coverage_{int(level*100)}pct_Raw": cov_raw,
                     f"Coverage_{int(level*100)}pct_Calibrated": cov_scaled})
    return pd.DataFrame(rows)

lofo_transfer = check_ood_coverage_transfer(lofo_preds_df, "Held_Out_Fluid")
loso_transfer = check_ood_coverage_transfer(loso_preds_df, "Held_Out_Study")

print("\n--- LOFO: does the validation-fit s help under fluid-level shift? ---")
print(lofo_transfer.to_string(index=False))
print("\n--- LOSO: does the validation-fit s help under study-level shift? ---")
print(loso_transfer.to_string(index=False))

lofo_transfer.to_excel(os.path.join(EXCEL_DIR, "final_lofo_ood_calibration_transfer.xlsx"), index=False)
loso_transfer.to_excel(os.path.join(EXCEL_DIR, "final_loso_ood_calibration_transfer.xlsx"), index=False)

print("\nIf coverage is still well below 0.95 for a fold even after scaling, that "
      "fold's shift is too large for a single global scalar -- report it honestly "
      "rather than inflating s further just to paper over it. See the split-conformal "
      "block below for a distribution-free alternative, and Tibshirani, Barber, Candes "
      "& Ramdas (2019), 'Conformal Prediction Under Covariate Shift,' NeurIPS 32, "
      "2530-2540, for a shift-aware (weighted) conformal extension if you need a "
      "rigorous guarantee on the OOD folds rather than just this diagnostic.")

# %%
def fit_split_conformal(model, X_cal, y_cal, alpha=0.05):
    """Normalized nonconformity scores in log-space: R_i = |y_i - mu_i| / sigma_i."""
    pred_log_cal = model.predict(X_cal)
    log_std_cal = model.pred_dist(X_cal).scale
    y_true_cal = y_cal.values if hasattr(y_cal, "values") else np.asarray(y_cal)

    scores = np.abs(y_true_cal - pred_log_cal) / log_std_cal
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(scores, q_level, method="higher")
    return q_hat

Q_HAT_95 = fit_split_conformal(final_model, X_val, y_val, alpha=0.05)
print(f"Split-conformal quantile q_hat (95%, fit on validation): {Q_HAT_95:.4f}")

def conformal_bounds(model, X_new, q_hat=Q_HAT_95):
    pred_log = model.predict(X_new)
    log_std = model.pred_dist(X_new).scale
    lower = np.exp(pred_log - q_hat * log_std)
    upper = np.exp(pred_log + q_hat * log_std)
    return np.exp(pred_log), lower, upper

_, test_lower_cp, test_upper_cp = conformal_bounds(final_model, X_test)
test_true_orig = np.exp(y_test.values)
empirical_cov_cp = np.mean((test_true_orig >= test_lower_cp) & (test_true_orig <= test_upper_cp))
print(f"Split-conformal empirical coverage on test_untouched (target 95%): {empirical_cov_cp:.4f}")
print("Compare this to the calibrated-Gaussian coverage saved in "
      "final_opt_test_untouched_calibrated_calibration_data.xlsx -- if they roughly "
      "agree, Solution A (Gaussian variance scaling) is an adequate, more interpretable "
      "fix; if conformal coverage is noticeably closer to 95%, the Gaussian shape "
      "assumption is the limiting factor and conformal bounds are the safer choice "
      "to report.")

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

# Directories
OUT_DIR = "publication_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
EXCEL_DIR = os.path.join(OUT_DIR, "excel_data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)

# 1. PLOT & EXPORT TRAIN, VALIDATION, AND TEST SPLITS
def plot_and_export_splits(X_split, y_split, split_name, fluid_labels):
    pred_log = final_model.predict(X_split)
    dist = final_model.pred_dist(X_split)
    
    pred_clipped = np.clip(pred_log, np.log(y_raw.min() * 0.1), np.log(y_raw.max() * 10.0))
    pred_orig = np.exp(pred_clipped)
    true_orig = np.exp(y_split)
    
    # Calculate mathematically correct asymmetric bounds
    log_std = dist.scale
    lower_bound = np.exp(pred_log - (1.96 * log_std))
    upper_bound = np.exp(pred_log + (1.96 * log_std))
    
    # Save Data & Error Data to Excel
    report_df = pd.DataFrame({
        "Fluid": fluid_labels.values,
        "True_Value": true_orig.values,
        "Predicted_Value": pred_orig,
        "Log_Standard_Deviation": log_std,
        "Lower_Bound_95CI": lower_bound,
        "Upper_Bound_95CI": upper_bound
    })
    report_df.to_excel(os.path.join(EXCEL_DIR, f"{split_name}_predictions_and_errors.xlsx"), index=False)
    
    # Calculate asymmetric error margins for matplotlib
    yerr_lower = pred_orig - lower_bound
    yerr_upper = upper_bound - pred_orig
    
    # Scatter Plot with True Asymmetric Error Bars
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(true_orig, pred_orig, yerr=[yerr_lower, yerr_upper], fmt='o', color='teal', ecolor='darkcyan',
                elinewidth=0.8, alpha=0.5, label="Prediction 95% CI")
    min_val, max_val = min(true_orig.min(), pred_orig.min()), max(true_orig.max(), pred_orig.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal Fit (1:1)")
    ax.set_xlabel("Measured Pressure Drop", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Pressure Drop", fontsize=11, fontweight="bold")
    ax.set_title(f"{split_name.capitalize()} Split: Error Bars (95% CI)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{split_name}_error_bars.png"), dpi=300)
    plt.close(fig)
    print(f"[{split_name.upper()}] Plot and Excel file saved.")

plot_and_export_splits(X_train, y_train, "train", f_train)
plot_and_export_splits(X_val, y_val, "validation", f_val)
plot_and_export_splits(X_test, y_test, "test_untouched", f_test)


# 2. PLOT & EXPORT LOSO AND LOFO EVALUATIONS
def export_and_plot_group_cv(group_labels, group_name_str):
    logo = LeaveOneGroupOut()
    summary_results = []
    all_preds = []
    
    for train_idx, test_idx in logo.split(X, y, groups=group_labels):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        w_tr = sample_weights[train_idx]
        group_val = group_labels.iloc[test_idx].iloc[0]
        
        # Fit model for this held-out group
        base_learner = RandomForestRegressor(n_estimators=15, max_depth=4, min_samples_leaf=5, max_features=0.9, random_state=42, n_jobs=None)
        # Using Dist=Normal as target is already manually logged
        model = NGBRegressor(Dist=Normal, Score=LogScore, Base=base_learner, n_estimators=400, learning_rate=0.01, random_state=42, verbose=False)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        pred_log = model.predict(X_te)
        dist = model.pred_dist(X_te)
        
        pred_clipped = np.clip(pred_log, np.log(y_raw.min() * 0.1), np.log(y_raw.max() * 10.0))
        pred_orig = np.exp(pred_clipped)
        true_orig = np.exp(y_te)
        
        # Correctly bound physical errors
        log_std = dist.scale
        lower_bound = np.exp(pred_log - (1.96 * log_std))
        upper_bound = np.exp(pred_log + (1.96 * log_std))
        
        mape = mean_absolute_percentage_error(true_orig, pred_orig) * 100
        r2 = r2_score(true_orig, pred_orig) if len(true_orig) > 1 else np.nan
        
        summary_results.append({f"Held_Out_{group_name_str}": group_val, "Samples": len(y_te), "MAPE_%": mape, "R2": r2})
        
        group_df = pd.DataFrame({
            f"Held_Out_{group_name_str}": [group_val] * len(true_orig),
            "True_Value": true_orig.values,
            "Predicted_Value": pred_orig,
            "Log_Standard_Deviation": log_std,
            "Lower_Bound_95CI": lower_bound,
            "Upper_Bound_95CI": upper_bound,
            "APE_%": np.abs(pred_orig - true_orig.values) / true_orig.values * 100
        })
        all_preds.append(group_df)

    # Save Summary & All Detailed Predictions/Errors to Excel
    res_df = pd.DataFrame(summary_results).sort_values(by="MAPE_%", ascending=False)
    res_df.to_excel(os.path.join(EXCEL_DIR, f"lo{group_name_str[0]}o_{group_name_str.lower()}_summary.xlsx"), index=False)
    
    combined_df = pd.concat(all_preds, ignore_index=True)
    combined_df.to_excel(os.path.join(EXCEL_DIR, f"lo{group_name_str[0]}o_{group_name_str.lower()}_all_predictions_and_errors.xlsx"), index=False)
    
    # Plotting Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['crimson' if m > 40 else 'royalblue' for m in res_df["MAPE_%"]]
    ax.bar(range(len(res_df)), res_df["MAPE_%"], color=colors, edgecolor="black")
    ax.axhline(40, color="darkred", linestyle="--", linewidth=1.5, label="40% Error Threshold")
    ax.set_xticks(range(len(res_df)))
    ax.set_xticklabels(res_df[f"Held_Out_{group_name_str}"], rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("MAPE (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Leave-One-{group_name_str}-Out Generalization MAPE", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"lo{group_name_str[0]}o_mape_by_{group_name_str.lower()}.png"), dpi=300)
    plt.close(fig)
    print(f"[{group_name_str.upper()}] LO{group_name_str[0]}O Bar plot and Excel reports saved.")

export_and_plot_group_cv(fluids, "Fluid")
export_and_plot_group_cv(studies, "Study")

# %%
# SHAP INTERPRETABILITY ANALYSIS (ON TRAIN/TESTED MODEL)

import shap
import matplotlib.ticker as ticker # Imported for precise axis tick control

print("\n--- Generating SHAP Analysis Plots for Final Model ---")

try:
    # Use the model-agnostic explainer on the full ensemble's predict function.
    # We use X_train as the background dataset to establish a baseline.
    explainer = shap.Explainer(final_model.predict, X_train)
    
    # Calculate SHAP values specifically for the untouched test set
    shap_values = explainer(X_test)
    
    # 1. SHAP Summary Plot (Beeswarm) - Shows impact and direction of features
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values.values, X_test, feature_names=SELECTED_FEATURES, show=False)
    
    # FIX: Force proper x-axis ticks to show the end values
    current_ax = plt.gca()
    current_ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    
    plt.title("SHAP Feature Importance (Untouched Test Set)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    # FIX: Save as both PNG (preview) and SVG (Fully editable in PowerPoint)
    fig.savefig(os.path.join(FIGURES_DIR, "shap_summary_plot.png"), dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, "shap_summary_plot.svg"), format='svg', transparent=True)
    plt.close(fig)
    
    # 2. SHAP Bar Plot - Shows global mean absolute importance
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values.values, X_test, feature_names=SELECTED_FEATURES, plot_type="bar", show=False)
    
    # FIX: Force proper x-axis ticks to show the end values on the bar chart
    current_ax = plt.gca()
    current_ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    
    plt.title("SHAP Global Feature Importance", fontsize=12, fontweight="bold")
    fig.tight_layout()
    
    # FIX: Save as both PNG (preview) and SVG (Fully editable in PowerPoint)
    fig.savefig(os.path.join(FIGURES_DIR, "shap_bar_plot.png"), dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, "shap_bar_plot.svg"), format='svg', transparent=True)
    plt.close(fig)
    
    print(f"SHAP plots successfully generated and saved to ./{FIGURES_DIR}/")

except Exception as e:
    print(f"SHAP generation failed. Error: {e}")

# %%
#SAVE TRAINED MODEL FOR EXTERNAL TESTING

import pickle
import sklearn
import ngboost
import platform
from datetime import datetime

MODEL_EXPORT_DIR = os.path.join(OUTPUT_DIR, "model_export")
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)

LOG_CLIP_MIN = np.log(y_raw.min() * 0.1)
LOG_CLIP_MAX = np.log(y_raw.max() * 10.0)

model_bundle = {
    "model": final_model,
    "feature_names": SELECTED_FEATURES,
    "target_name": TARGET,
    "target_transform": "log",
    "log_clip_bounds": (LOG_CLIP_MIN, LOG_CLIP_MAX),
    "ci_z_value": 1.96,
   
    "calibration_s": CALIBRATION_S,
    "conformal_q95": Q_HAT_95,
    "training_info": {
        "n_train_samples": len(X_train),
        "date_saved": datetime.now().strftime("%Y-%m-%d"),
        "ngboost_version": ngboost.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    },
    "notes": ("NGBRegressor (Normal dist) with RandomForestRegressor base learner. "
              "Predicts log(target); use predict_with_uncertainty() to get mean + "
              "CALIBRATED 95% CI back in original units (raw model sigma is "
              "overconfident -- see calibration_s / conformal_q95).")}

MODEL_PATH = os.path.join(MODEL_EXPORT_DIR, "ngboost_pressure_gradient_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"Model bundle saved to: {MODEL_PATH}")
print(f"Required input features (in order): {SELECTED_FEATURES}")
print(f"ngboost=={ngboost.__version__}, scikit-learn=={sklearn.__version__} "
      f"— give reviewers these exact versions to avoid pickle-compatibility issues.")

# %%
# LOAD MODEL & PREDICT ON LITERATURE DATA

import pickle
import glob
import numpy as np
import pandas as pd

# ---- Load the saved model bundle ----
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_export", "ngboost_pressure_gradient_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

loaded_model = model_bundle["model"]
FEATURES = model_bundle["feature_names"]
LOG_CLIP_MIN, LOG_CLIP_MAX = model_bundle["log_clip_bounds"]
Z = model_bundle["ci_z_value"]
CAL_S = model_bundle.get("calibration_s", 1.0)
CAL_Q95 = model_bundle.get("conformal_q95", None)

print("Loaded model expects these features, in this order:")
print(FEATURES)
print(f"Applying persisted calibration factor s = {CAL_S:.4f} to the 95% CI bounds "
      f"(point predictions are unaffected).")

def predict_with_uncertainty(X_new: pd.DataFrame) -> pd.DataFrame:
    """
    X_new: DataFrame with at least the columns in FEATURES.
    Returns mean prediction + CALIBRATED 95% CI in original (untransformed)
    target units, plus the raw (uncalibrated) std and, if available, the
    split-conformal bounds as a distribution-free cross-check.
    """
    missing = [c for c in FEATURES if c not in X_new.columns]
    if missing:
        raise KeyError(f"Input data is missing required columns: {missing}")

    X_ordered = X_new[FEATURES].reset_index(drop=True)

    pred_log = loaded_model.predict(X_ordered)
    dist = loaded_model.pred_dist(X_ordered)
    log_std = dist.scale
    calibrated_std = CAL_S * log_std

    pred_log_clipped = np.clip(pred_log, LOG_CLIP_MIN, LOG_CLIP_MAX)
    pred_mean = np.exp(pred_log_clipped)
    lower_95 = np.exp(pred_log - Z * calibrated_std)
    upper_95 = np.exp(pred_log + Z * calibrated_std)

    result = pd.DataFrame({
        "Predicted_Mean": pred_mean,
        "Lower_Bound_95CI": lower_95,
        "Upper_Bound_95CI": upper_95,
        "Raw_Log_Standard_Deviation": log_std,
        "Calibration_Factor_s": CAL_S,
        "Calibrated_Log_Standard_Deviation": calibrated_std,})

    if CAL_Q95 is not None:
        result["Conformal_Lower_95CI"] = np.exp(pred_log - CAL_Q95 * log_std)
        result["Conformal_Upper_95CI"] = np.exp(pred_log + CAL_Q95 * log_std)

    return result

# ---- Find and load the literature file 
lit_matches = glob.glob("Literature.*")

if not lit_matches:
    raise FileNotFoundError(
        f"No file named Literature.* found in {os.getcwd()}. "
        f"Files present: {os.listdir('.')}")

LITERATURE_PATH = lit_matches[0]
print(f"Loading literature data from: {LITERATURE_PATH}")

lit_df = pd.read_excel(LITERATURE_PATH, sheet_name="Sheet1")

# ---- Run to get predictions ----
lit_results = predict_with_uncertainty(lit_df)

# Just keeping any identifying columns (fluid, study, source, etc.) alongside the predictions
id_cols = [c for c in lit_df.columns if c not in FEATURES]
lit_output = pd.concat([lit_df[id_cols].reset_index(drop=True), lit_results], axis=1)

lit_output_path = os.path.join(EXCEL_DIR, "literature_predictions_with_uncertainty.xlsx")
lit_output.to_excel(lit_output_path, index=False)

print(f"Literature predictions saved to: {lit_output_path}")
lit_output.head()

# %%
#SANITY CHECK 

check_results = predict_with_uncertainty(X_test)

original_pred_log = final_model.predict(X_test)
original_mean = np.exp(np.clip(original_pred_log, LOG_CLIP_MIN, LOG_CLIP_MAX))

max_diff = np.max(np.abs(check_results["Predicted_Mean"].values - original_mean))
print(f"Max absolute difference between original and reloaded predictions: {max_diff:.10f}")
assert max_diff < 1e-8, "Reloaded model does not match original — investigate!"
print("Round-trip check passed: reloaded model produces identical predictions.")
