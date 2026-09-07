"""
Final model: Ensemble LightGBM + XGBoost with interaction features
- Two-quantile strategy (P50 + P75)
- Sample weighting for live days
- 6 interaction features with is_live_day
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    r2_score, median_absolute_error
)
import shap


# ========================  Data Pipeline  ========================

def build_features(df):
    """Feature engineering + interaction features"""
    from feature_engineer import LiveStreamFeatureEngineer

    eng = LiveStreamFeatureEngineer()
    X, y = eng.fit_transform(df)

    # Add is_live_day + interaction features
    il = df['is_live_day'].values[:len(X)]
    X['is_live_day'] = il
    X['disc_x_live'] = X['discount'] * il
    X['traf_x_live'] = X['platform_traffic'] * il
    X['anch_x_live'] = X['anchor_fans'] * il
    X['comp_x_live'] = X['competitor_discount'] * il
    X['lag1_x_live'] = X['sales_lag_1'] * il
    X['rol7_x_live'] = X['sales_roll7_mean'] * il

    cat_feats = eng.categorical_features_ + ['is_live_day']
    return X, y, cat_feats


# ========================  Training  ========================

def train_ensemble(X_train, y_train, X_test, y_test, cat_feats):
    """Train LGB P50 + LGB P75 + XGB P50 ensemble"""

    # Sample weights: live days 8x
    w = np.ones(len(y_train))
    w[X_train['is_live_day'] == 1] = 8.0

    cb_lgb = [lgb.early_stopping(50), lgb.log_evaluation(0)]

    # LGB P50
    print("Training LGB P50...")
    lgb50 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.5, metric='quantile',
        num_leaves=31, learning_rate=0.02, n_estimators=1000,
        max_depth=6, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    )
    lgb50.fit(X_train, y_train, sample_weight=w,
              eval_set=[(X_test, y_test)],
              categorical_feature=cat_feats, callbacks=cb_lgb)

    # LGB P75
    print("Training LGB P75...")
    lgb75 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.75, metric='quantile',
        num_leaves=15, learning_rate=0.02, n_estimators=1000,
        max_depth=5, min_child_samples=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    )
    lgb75.fit(X_train, y_train, sample_weight=w,
              eval_set=[(X_test, y_test)],
              categorical_feature=cat_feats, callbacks=cb_lgb)

    # XGB P50
    print("Training XGB P50...")
    xgb50 = xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=0.5,
        n_estimators=500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=0,
    )
    xgb50.fit(X_train, y_train, sample_weight=w)

    print("Training complete.\n")
    return lgb50, lgb75, xgb50


def predict_ensemble(models, X):
    """Ensemble prediction:
       Non-live: 0.5*P50_lgb + 0.5*P50_xgb
       Live:     0.3*P50_lgb + 0.3*P50_xgb + 0.4*P75_lgb
    """
    lgb50, lgb75, xgb50 = models
    y50_l = lgb50.predict(X)
    y75_l = lgb75.predict(X)
    y50_x = xgb50.predict(X)

    is_live = X['is_live_day'].values == 1
    yp = np.where(
        is_live,
        0.3 * y50_l + 0.3 * y50_x + 0.4 * y75_l,
        0.5 * y50_l + 0.5 * y50_x,
    )
    return yp


# ========================  Evaluation  ========================

def evaluate(y_true, y_pred):
    m = y_true.values > 100
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MdAE': median_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true[m], y_pred[m]) * 100,
        'R2': r2_score(y_true, y_pred),
        'DirAcc': (np.sign(np.diff(np.r_[y_true.iloc[0], y_true])) ==
                    np.sign(np.diff(np.r_[y_pred[0], y_pred]))).mean() * 100,
    }


# ========================  5 Plots  ========================

def plot_data_overview(df, save_path='fig1_data_overview.png'):
    fig, ax = plt.subplots(figsize=(14, 5))
    live = df[df['is_live_day'] == 1]
    non = df[df['is_live_day'] == 0]
    ax.plot(non['date'], non['sales'], color='gray', alpha=0.6, linewidth=0.8)
    ax.scatter(live['date'], live['sales'], color='red', s=50, zorder=5,
               label=f'Live Day ({len(live)} days)')
    ax.set_title('Fig 1: Sales Data - Live Days Highlighted', fontsize=14)
    ax.set_xlabel('Date'); ax.set_ylabel('Sales')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_prediction_vs_actual(y_test, y_pred, mape, save_path='fig2_prediction.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Scatter (MAPE={mape:.1f}%)')
    axes[0].legend(['Perfect']); axes[0].grid(True, alpha=0.3)

    x = range(len(y_test))
    axes[1].plot(x, y_test.values, 'b-', label='Actual', lw=1.5, alpha=0.8)
    axes[1].plot(x, y_pred, 'orange', label='Predicted', lw=1.5, alpha=0.8)
    axes[1].set_xlabel('Test Sample'); axes[1].set_ylabel('Sales')
    axes[1].set_title('Time Series: Actual vs Predicted')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle('Fig 2: Ensemble Prediction Results', fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_residual_distribution(y_test, y_pred, save_path='fig3_residual.png'):
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuals, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(0, color='red', linestyle='--', lw=2,
                    label=f'Mean={residuals.mean():.0f}')
    axes[0].set_xlabel('Residual'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual')
    axes[1].set_title('Residual vs Predicted')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('Fig 3: Residual Analysis', fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_importance(lgb50, X_test, save_path='fig4_shap.png'):
    explainer = shap.TreeExplainer(lgb50)
    shap_values = explainer.shap_values(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-10:][::-1]
    top_feat = X_test.columns[top_idx]
    top_vals = mean_abs[top_idx]

    ax = axes[0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 10))[::-1]
    ax.barh(range(10), top_vals[::-1], color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_feat[::-1])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Top 10 Feature Importance (SHAP)')

    ax = axes[1]
    X_top = X_test[top_feat[:8]].values
    sh_top = shap_values[:, top_idx[:8]]
    for j in range(8):
        fv = X_top[:, j]; sv = sh_top[:, j]
        norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
        ax.scatter(sv, [7-j] * len(sv), c=plt.cm.RdYlBu(norm),
                   alpha=0.5, s=20, edgecolors='none')
    ax.set_yticks(range(8))
    ax.set_yticklabels(top_feat[:8][::-1])
    ax.axvline(0, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('SHAP Value'); ax.set_title('SHAP Distribution (Top 8)')
    fig.suptitle('Fig 4: SHAP Feature Analysis', fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_supply_chain_plan(y_test, y_pred, save_path='fig5_supply_plan.png'):
    residuals = y_test.values - y_pred
    sigma = np.std(residuals)
    n = min(8, len(y_test))
    ya = y_test.values[-n:]; yp = y_pred[-n:]
    lo = yp - 1.28 * sigma; hi = yp + 1.28 * sigma

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n); w = 0.22
    ax.bar(x - w, np.maximum(lo, 0), w, color='#2ecc71', label='Safe (Own Line)', alpha=0.85)
    ax.bar(x, yp, w, color='#3498db', label='Expected (Own+Outsource)', alpha=0.85)
    ax.bar(x + w, np.maximum(hi, 0), w, color='#e67e22', label='Sprint (Materials Only)', alpha=0.85)
    ax.scatter(x, ya, color='red', s=80, zorder=5, marker='*', label='Actual', edgecolors='darkred')
    ax.set_xticks(x); ax.set_xticklabels([f'S{i+1}' for i in range(n)])
    ax.set_ylabel('Sales'); ax.set_title('Fig 5: Three-Tier Inventory Strategy', fontsize=14)
    ax.legend(loc='upper left'); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


# ============================  Main  ============================

if __name__ == '__main__':
    # 1. Load + features
    df = pd.read_csv('../data/synthetic_sales_data.csv', parse_dates=['date'])
    X, y, cat_feats = build_features(df)

    # 2. Time-series split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Live days in test: {(X_test['is_live_day']==1).sum()}")

    # 3. Train ensemble
    models = train_ensemble(X_train, y_train, X_test, y_test, cat_feats)

    # 4. Predict
    y_pred = predict_ensemble(models, X_test)

    # 5. Evaluate
    m_all = evaluate(y_test, y_pred)
    m_nl = evaluate(y_test[X_test['is_live_day']==0], y_pred[X_test['is_live_day']==0])
    m_l = evaluate(y_test[X_test['is_live_day']==1], y_pred[X_test['is_live_day']==1])

    print("=" * 60)
    print(f"{'Metric':<15} {'All':>10} {'Non-Live':>10} {'Live':>10}")
    print("-" * 60)
    for k in ['MAE', 'MdAE', 'MAPE', 'R2', 'DirAcc']:
        fmt = '.1f' if k == 'MAPE' or k == 'DirAcc' else '.0f'
        fmt_r2 = '.3f' if k == 'R2' else fmt
        f = fmt_r2
        print(f"{k:<15} {m_all[k]:>10{f}} {m_nl[k]:>10{f}} {m_l[k]:>10{f}}")
    print("=" * 60)

    # 6. Generate 5 plots
    print("\nGenerating plots...")
    plot_data_overview(df)
    plot_prediction_vs_actual(y_test, y_pred, m_all['MAPE'])
    plot_residual_distribution(y_test, y_pred)
    plot_shap_importance(models[0], X_test)
    plot_supply_chain_plan(y_test, y_pred)

    # 7. Feature importance
    print("\n" + "=" * 60)
    print("LGB P50 Feature Importance Top 10")
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance': models[0].feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    for _, r in imp.iterrows():
        print(f"  {r['feature']:35s} = {r['importance']:>8.0f}")

    # 8. Three-tier examples
    print("\n" + "=" * 60)
    print("Three-Tier Strategy (last 5 test samples)")
    sigma = np.std(y_test.values - y_pred)
    print(f"  Residual Std: {sigma:.0f}")
    for i in range(min(5, len(y_test))):
        idx = -(i+1)
        pred = y_pred[idx]; actual = y_test.values[idx]
        safe = int(max(0, pred - 1.28 * sigma))
        sprint = int(max(0, pred + 1.28 * sigma))
        is_live = X_test['is_live_day'].values[idx]
        tag = '[LIVE]' if is_live else ''
        print(f"  S{len(y_test)-i} {tag:6s} Actual={actual:>7.0f}  "
              f"Predict={int(pred):>7.0f}  "
              f"Safe={safe:>6.0f}  Sprint={sprint:>6.0f}")

    print("\nDone!")
