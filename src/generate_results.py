"""
Generate comprehensive results including SPY benchmark and all visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from portfolio_optimizer import PortfolioOptimizer, PortfolioBacktester
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*80)
print("GENERATING COMPREHENSIVE RESULTS")
print("="*80)

# Load data
print("\nLoading data...")
features_data = pd.read_csv('../data/features_data.csv', index_col=0, parse_dates=True)

# Prepare portfolio returns
portfolio_assets = ['WTI_RETURN', 'XLE_RETURN', 'XOM_RETURN', 'CVX_RETURN', 'IEF_RETURN']
portfolio_returns = features_data[portfolio_assets].copy()
portfolio_returns.columns = ['WTI', 'XLE', 'XOM', 'CVX', 'IEF']
portfolio_returns = portfolio_returns.dropna()

# SPY benchmark returns
spy_returns = features_data['SPY_RETURN'].reindex(portfolio_returns.index).fillna(0)

# Macro scores
macro_scores = features_data['MACRO_SCORE'].reindex(portfolio_returns.index).fillna(0)

print(f"Data loaded: {len(portfolio_returns)} trading days")

# Initialize backtester
backtester = PortfolioBacktester(
    portfolio_returns,
    rebalance_freq='M',
    transaction_cost=0.001
)

optimizer = PortfolioOptimizer(portfolio_returns, risk_free_rate=0.02)

# Run all backtests
print("\n" + "="*80)
print("RUNNING BACKTESTS (Including SPY Benchmark)")
print("="*80)

strategies = {}

# 0. SPY Benchmark
print("\n[1/5] SPY Buy-and-Hold Benchmark...")
spy_value = (1 + spy_returns).cumprod()
strategies['SPY_Benchmark'] = pd.DataFrame({
    'portfolio_return': spy_returns,
    'portfolio_value': spy_value
})

# 1. Equal Weight
print("[2/5] Equal Weight Strategy...")
strategies['Equal_Weight'] = backtester.backtest_strategy(
    'Equal Weight',
    lambda: optimizer.equal_weight()
)

# 2. Volatility Parity
print("[3/5] Volatility Parity Strategy...")
strategies['Volatility_Parity'] = backtester.backtest_strategy(
    'Volatility Parity',
    lambda: optimizer.volatility_parity(lookback=126)
)

# 3. Mean-Variance
print("[4/5] Mean-Variance Optimization...")
strategies['Mean_Variance'] = backtester.backtest_strategy(
    'Mean-Variance',
    lambda: optimizer.mean_variance_optimization(lookback=126)
)

# 4. Macro-Adaptive
print("[5/5] Macro-Adaptive Strategy...")
strategies['Macro_Adaptive'] = backtester.backtest_strategy(
    'Macro-Adaptive',
    optimizer.macro_adaptive_weights,
    macro_scores=macro_scores,
    energy_indices=[0, 1, 2, 3],
    safe_indices=[4],
    tilt_magnitude=0.20
)

# Calculate performance metrics
print("\n" + "="*80)
print("CALCULATING PERFORMANCE METRICS")
print("="*80)

metrics_dict = {}
for name, results in strategies.items():
    metrics = backtester.calculate_performance_metrics(results, risk_free_rate=0.02)
    metrics_dict[name] = metrics

metrics_df = pd.DataFrame(metrics_dict).T
print("\n" + metrics_df.round(2).to_string())

# Save metrics
metrics_df.to_csv('../data/performance_metrics_complete.csv')
print("\n✓ Saved: ../data/performance_metrics_complete.csv")

# Save individual results
for name, results in strategies.items():
    filename = f'../data/backtest_{name.lower()}.csv'
    results.to_csv(filename)
    print(f"✓ Saved: {filename}")

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING CHARTS")
print("="*80)

# Chart 1: Cumulative Performance
print("\n[1/7] Cumulative Performance Chart...")
fig, ax = plt.subplots(figsize=(14, 7))

colors = ['black', 'steelblue', 'coral', 'mediumseagreen', 'purple']
for i, (name, results) in enumerate(strategies.items()):
    label = name.replace('_', ' ')
    ax.plot(results.index, results['portfolio_value'],
            label=label, linewidth=2.5 if name == 'SPY_Benchmark' else 2,
            linestyle='--' if name == 'SPY_Benchmark' else '-',
            color=colors[i % len(colors)])

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Performance Comparison (Jan 2015 - Nov 2025)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(1, color='gray', linestyle=':', linewidth=1)
plt.tight_layout()
plt.savefig('../data/cumulative_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ../data/cumulative_performance.png")

# Chart 2: Rolling Sharpe and Volatility
print("[2/7] Rolling Metrics (Sharpe & Volatility)...")
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax = axes[0]
for name, results in strategies.items():
    returns = results['portfolio_return']
    rolling_sharpe = (returns.rolling(252).mean() / returns.rolling(252).std()) * np.sqrt(252)
    ax.plot(results.index, rolling_sharpe, label=name.replace('_', ' '), linewidth=2)

ax.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linestyle='--', linewidth=1)

ax = axes[1]
for name, results in strategies.items():
    returns = results['portfolio_return']
    rolling_vol = returns.rolling(252).std() * np.sqrt(252) * 100
    ax.plot(results.index, rolling_vol, label=name.replace('_', ' '), linewidth=2)

ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_ylabel('Volatility (%)', fontsize=11, fontweight='bold')
ax.set_title('Rolling 1-Year Volatility (Annualized)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/rolling_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ../data/rolling_metrics.png")

# Chart 3: Drawdown Analysis
print("[3/7] Drawdown Analysis...")
fig, ax = plt.subplots(figsize=(14, 7))

for name, results in strategies.items():
    values = results['portfolio_value']
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100
    ax.plot(results.index, drawdown, label=name.replace('_', ' '), linewidth=2)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax.set_title('Drawdown from Peak', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.fill_between(ax.get_xlim(), -60, 0, alpha=0.1, color='red')
plt.tight_layout()
plt.savefig('../data/drawdown_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ../data/drawdown_analysis.png")

# Chart 4: Risk-Return Scatter
print("[4/7] Risk-Return Scatter Plot...")
fig, ax = plt.subplots(figsize=(12, 8))

colors_scatter = plt.cm.Set2(np.linspace(0, 1, len(metrics_df)))

for i, (name, row) in enumerate(metrics_df.iterrows()):
    marker = 's' if name == 'SPY_Benchmark' else 'o'
    size = 300 if name == 'SPY_Benchmark' else 200
    ax.scatter(row['Annual Volatility (%)'], row['Annual Return (%)'],
              s=size, alpha=0.7, c=[colors_scatter[i]], edgecolors='black',
              linewidth=2, label=name.replace('_', ' '), marker=marker)

    ax.annotate(name.replace('_', ' '),
                xy=(row['Annual Volatility (%)'], row['Annual Return (%)']),
                xytext=(8, 8), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

ax.set_xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
ax.set_title('Risk-Return Profile (2015-2025)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('../data/risk_return_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ../data/risk_return_scatter.png")

# Chart 5: Performance Comparison Bars
print("[5/7] Performance Comparison Bar Charts...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax = axes[0, 0]
metrics_df['Annual Return (%)'].sort_values().plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
ax.set_xlabel('Annual Return (%)', fontsize=11)
ax.set_title('Annual Return Comparison', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

ax = axes[0, 1]
metrics_df['Sharpe Ratio'].sort_values().plot(kind='barh', ax=ax, color='coral', edgecolor='black')
ax.set_xlabel('Sharpe Ratio', fontsize=11)
ax.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

ax = axes[1, 0]
metrics_df['Max Drawdown (%)'].sort_values(ascending=False).plot(kind='barh', ax=ax, color='crimson', edgecolor='black')
ax.set_xlabel('Max Drawdown (%)', fontsize=11)
ax.set_title('Maximum Drawdown (Less Negative = Better)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

ax = axes[1, 1]
metrics_df['Sortino Ratio'].sort_values().plot(kind='barh', ax=ax, color='mediumseagreen', edgecolor='black')
ax.set_xlabel('Sortino Ratio', fontsize=11)
ax.set_title('Sortino Ratio Comparison', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig('../data/performance_comparison_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ../data/performance_comparison_bars.png")

# Chart 6: Asset Correlation Heatmap
print("[6/7] Asset Correlation Heatmap...")
return_cols = ['WTI_RETURN', 'XLE_RETURN', 'XOM_RETURN', 'CVX_RETURN', 'IEF_RETURN', 'SPY_RETURN']
asset_corr = features_data[return_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(asset_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Asset Return Correlations', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../data/asset_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ../data/asset_correlation_heatmap.png")

# Chart 7: Return Distributions
print("[7/7] Return Distribution Histograms...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (name, results) in enumerate(strategies.items()):
    if i < 6:
        ax = axes[i]
        returns = results['portfolio_return'] * 100

        ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
        ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {returns.mean():.3f}%')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)

        ax.set_xlabel('Daily Return (%)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../data/return_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: ../data/return_distributions.png")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("CREATING SUMMARY TABLE")
print("="*80)

summary_table = metrics_df[['Total Return (%)', 'Annual Return (%)', 'Annual Volatility (%)',
                            'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)']].copy()

summary_table['Return Rank'] = summary_table['Annual Return (%)'].rank(ascending=False).astype(int)
summary_table['Sharpe Rank'] = summary_table['Sharpe Ratio'].rank(ascending=False).astype(int)
summary_table['DD Rank'] = summary_table['Max Drawdown (%)'].rank(ascending=False).astype(int)

print("\n" + summary_table.round(2).to_string())

summary_table.to_csv('../data/summary_table.csv')
print("\n✓ Saved: ../data/summary_table.csv")

print("\n" + "="*80)
print("✅ ALL RESULTS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nFiles created in ../data/:")
print("  CSV Files:")
print("    - performance_metrics_complete.csv")
print("    - summary_table.csv")
print("    - backtest_spy_benchmark.csv")
print("    - backtest_equal_weight.csv")
print("    - backtest_volatility_parity.csv")
print("    - backtest_mean_variance.csv")
print("    - backtest_macro_adaptive.csv")
print("\n  Chart Files:")
print("    - cumulative_performance.png")
print("    - rolling_metrics.png")
print("    - drawdown_analysis.png")
print("    - risk_return_scatter.png")
print("    - performance_comparison_bars.png")
print("    - asset_correlation_heatmap.png")
print("    - return_distributions.png")
print("\n" + "="*80)
