"""
Comprehensive Analysis - All Requirements
==========================================
Complete analysis including:
1. 25-year backtesting (1999-2024, actual: 2002-2024 due to data availability)
2. Monte Carlo simulation (1000 scenarios)
3. Walk-forward backtesting (5-year train, 1-year test)
4. Fee structure analysis (management 1-4%, performance 5-25%)
5. Explicit Alpha and Beta calculations
6. Business viability assessment

Author: Vrishani Shah
Course: MSDS 451 - Financial Engineering
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from portfolio_optimizer import PortfolioOptimizer, PortfolioBacktester
from monte_carlo import MonteCarloSimulator
from walk_forward import WalkForwardBacktester
from fee_analysis import FeeAnalyzer, calculate_alpha_beta
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*80)
print("COMPREHENSIVE ANALYSIS - ALL REQUIREMENTS")
print("="*80)
print("\nRequirements Covered:")
print("  ✓ 25-year historical period (actual: 2002-2024)")
print("  ✓ Monte Carlo simulation (robustness testing)")
print("  ✓ Walk-forward backtesting (avoid overfitting)")
print("  ✓ Fee structure analysis (management + performance)")
print("  ✓ Alpha & Beta calculations")
print("  ✓ Business viability assessment")
print("="*80)

# ==============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ==============================================================================

print("\n" + "="*80)
print("STEP 1: DATA PREPARATION")
print("="*80)

# Load feature-engineered data
features_data = pd.read_csv('../data/features_data.csv', index_col=0, parse_dates=True)

# Prepare portfolio returns
portfolio_assets = ['WTI_RETURN', 'XLE_RETURN', 'XOM_RETURN', 'CVX_RETURN', 'IEF_RETURN']
portfolio_returns = features_data[portfolio_assets].copy()
portfolio_returns.columns = ['WTI', 'XLE', 'XOM', 'CVX', 'IEF']
portfolio_returns = portfolio_returns.dropna()

# SPY benchmark
spy_returns = features_data['SPY_RETURN'].reindex(portfolio_returns.index).fillna(0)

# Macro scores
macro_scores = features_data['MACRO_SCORE'].reindex(portfolio_returns.index).fillna(0)

print(f"\nData loaded:")
print(f"  Period: {portfolio_returns.index[0].date()} to {portfolio_returns.index[-1].date()}")
print(f"  Trading days: {len(portfolio_returns)}")
print(f"  Years: {len(portfolio_returns) / 252:.1f}")
print(f"  Assets: {list(portfolio_returns.columns)}")

# ==============================================================================
# STEP 2: STANDARD BACKTESTING (Full Period)
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: FULL-PERIOD BACKTESTING")
print("="*80)

backtester = PortfolioBacktester(portfolio_returns, rebalance_freq='M', transaction_cost=0.001)
optimizer = PortfolioOptimizer(portfolio_returns, risk_free_rate=0.02)

strategies = {}

# SPY Benchmark
print("\n[1/5] SPY Benchmark...")
spy_value = (1 + spy_returns).cumprod()
strategies['SPY_Benchmark'] = pd.DataFrame({
    'portfolio_return': spy_returns,
    'portfolio_value': spy_value
})

# Equal Weight
print("[2/5] Equal Weight...")
strategies['Equal_Weight'] = backtester.backtest_strategy(
    'Equal Weight', lambda: optimizer.equal_weight()
)

# Volatility Parity
print("[3/5] Volatility Parity...")
strategies['Volatility_Parity'] = backtester.backtest_strategy(
    'Volatility Parity', lambda: optimizer.volatility_parity(lookback=126)
)

# Mean-Variance
print("[4/5] Mean-Variance...")
strategies['Mean_Variance'] = backtester.backtest_strategy(
    'Mean-Variance', lambda: optimizer.mean_variance_optimization(lookback=126)
)

# Macro-Adaptive
print("[5/5] Macro-Adaptive...")
strategies['Macro_Adaptive'] = backtester.backtest_strategy(
    'Macro-Adaptive',
    optimizer.macro_adaptive_weights,
    macro_scores=macro_scores,
    energy_indices=[0, 1, 2, 3],
    safe_indices=[4],
    tilt_magnitude=0.20
)

# Calculate performance metrics
metrics_dict = {}
for name, results in strategies.items():
    metrics = backtester.calculate_performance_metrics(results, risk_free_rate=0.02)
    metrics_dict[name] = metrics

metrics_df = pd.DataFrame(metrics_dict).T

print("\n" + metrics_df.round(2).to_string())
metrics_df.to_csv('../data/full_period_metrics.csv')

# ==============================================================================
# STEP 3: ALPHA & BETA CALCULATIONS
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: ALPHA & BETA ANALYSIS")
print("="*80)

alpha_beta_results = {}

for name, results in strategies.items():
    if name != 'SPY_Benchmark':
        ab = calculate_alpha_beta(results['portfolio_return'], spy_returns, risk_free_rate=0.02)
        alpha_beta_results[name] = ab

        print(f"\n{name}:")
        print(f"  Alpha (annual): {ab['alpha_annual']:.2f}%")
        print(f"  Beta: {ab['beta']:.2f}")
        print(f"  Correlation with SPY: {ab['correlation']:.2f}")
        print(f"  Tracking Error: {ab['tracking_error']:.2f}%")

alpha_beta_df = pd.DataFrame(alpha_beta_results).T
alpha_beta_df.to_csv('../data/alpha_beta_analysis.csv')

# ==============================================================================
# STEP 4: MONTE CARLO SIMULATION
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: MONTE CARLO SIMULATION")
print("="*80)

mc_simulator = MonteCarloSimulator(portfolio_returns, n_simulations=1000)

# Generate scenarios
print("\nGenerating scenarios...")
scenarios_normal = mc_simulator.generate_scenarios(n_periods=252, method='multivariate_normal')
scenarios_bootstrap = mc_simulator.generate_scenarios(n_periods=252, method='bootstrap')

# Test Equal Weight strategy
def equal_weight_mc(returns):
    return np.ones(len(returns.columns)) / len(returns.columns)

print("\nTesting on normal scenarios...")
mc_results_normal = mc_simulator.test_strategy(equal_weight_mc, scenarios_normal)

print("\nTesting on bootstrap scenarios...")
mc_results_bootstrap = mc_simulator.test_strategy(equal_weight_mc, scenarios_bootstrap)

# Summarize
mc_summary_normal = mc_simulator.summarize_results(mc_results_normal)
mc_summary_bootstrap = mc_simulator.summarize_results(mc_results_bootstrap)

print("\n" + "="*60)
print("Monte Carlo Summary (Multivariate Normal)")
print("="*60)
print(mc_summary_normal.round(2).to_string())

print("\n" + "="*60)
print("Monte Carlo Summary (Bootstrap)")
print("="*60)
print(mc_summary_bootstrap.round(2).to_string())

mc_results_normal.to_csv('../data/monte_carlo_normal_results.csv', index=False)
mc_results_bootstrap.to_csv('../data/monte_carlo_bootstrap_results.csv', index=False)
mc_summary_normal.to_csv('../data/monte_carlo_normal_summary.csv')
mc_summary_bootstrap.to_csv('../data/monte_carlo_bootstrap_summary.csv')

# ==============================================================================
# STEP 5: WALK-FORWARD BACKTESTING
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: WALK-FORWARD BACKTESTING")
print("="*80)

wf_backtester = WalkForwardBacktester(
    portfolio_returns,
    train_period_years=5,
    test_period_years=1,
    step_years=1
)

# Test Equal Weight
def equal_weight_wf(train_data):
    return np.ones(len(train_data.columns)) / len(train_data.columns)

wf_results = wf_backtester.backtest_strategy(equal_weight_wf)
wf_aggregate = wf_backtester.aggregate_results(wf_results)

print("\n" + "="*60)
print("Walk-Forward Results")
print("="*60)
for key, value in wf_aggregate.items():
    print(f"  {key}: {value:.2f}")

wf_results.to_csv('../data/walk_forward_results.csv', index=False)
pd.DataFrame([wf_aggregate]).to_csv('../data/walk_forward_summary.csv', index=False)

# ==============================================================================
# STEP 6: FEE STRUCTURE ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("STEP 6: FEE STRUCTURE ANALYSIS")
print("="*80)

# Analyze Equal Weight strategy with different fees
eq_weight_returns = strategies['Equal_Weight']['portfolio_return']
fee_analyzer = FeeAnalyzer(eq_weight_returns, spy_returns)

fee_scenarios = [
    (0, 0, "No Fees (Gross Returns)"),
    (1, 0, "1% Management Only"),
    (2, 0, "2% Management Only"),
    (3, 0, "3% Management Only"),
    (4, 0, "4% Management Only"),
    (2, 20, "2% Mgmt + 20% Performance"),
    (2, 25, "2% Mgmt + 25% Performance"),
]

fee_impacts = []

print("\nFee Impact on Equal Weight Strategy:")
for mgmt, perf, label in fee_scenarios:
    impact = fee_analyzer.calculate_fee_impact(mgmt, perf)
    fee_impacts.append(impact)

    print(f"\n{label}:")
    print(f"  Gross Annual Return: {impact['gross_annual_return_pct']:.2f}%")
    print(f"  Net Annual Return: {impact['net_annual_return_pct']:.2f}%")
    print(f"  Fee Drag: {impact['annual_fee_drag_pct']:.2f}%")

fee_impacts_df = pd.DataFrame(fee_impacts)
fee_impacts_df.to_csv('../data/fee_impact_analysis.csv', index=False)

# Full sensitivity analysis
sensitivity = fee_analyzer.fee_sensitivity_analysis(
    management_fees=[0, 1, 2, 3, 4],
    performance_fees=[0, 5, 10, 15, 20, 25]
)
sensitivity.to_csv('../data/fee_sensitivity_grid.csv', index=False)

# ==============================================================================
# STEP 7: BUSINESS VIABILITY ASSESSMENT
# ==============================================================================

print("\n" + "="*80)
print("STEP 7: BUSINESS VIABILITY ASSESSMENT")
print("="*80)

# Get SPY metrics
spy_annual_return = metrics_df.loc['SPY_Benchmark', 'Annual Return (%)']
spy_sharpe = metrics_df.loc['SPY_Benchmark', 'Sharpe Ratio']

# Get best energy strategy metrics (net of 2% mgmt + 20% perf fees)
best_strategy = 'Macro_Adaptive'
best_gross_return = metrics_df.loc[best_strategy, 'Annual Return (%)']

# Apply typical fee structure (2% + 20%)
best_returns = strategies[best_strategy]['portfolio_return']
best_fee_analyzer = FeeAnalyzer(best_returns, spy_returns)
best_net_impact = best_fee_analyzer.calculate_fee_impact(2.0, 20.0)
best_net_return = best_net_impact['net_annual_return_pct']

print("\n" + "="*60)
print("INVESTMENT COMPARISON")
print("="*60)
print(f"\nSPY Benchmark (Buy & Hold, No Fees):")
print(f"  Annual Return: {spy_annual_return:.2f}%")
print(f"  Sharpe Ratio: {spy_sharpe:.2f}")

print(f"\nBest Energy Strategy ({best_strategy}):")
print(f"  Gross Annual Return: {best_gross_return:.2f}%")
print(f"  Net Annual Return (after 2% mgmt + 20% perf fees): {best_net_return:.2f}%")
print(f"  Alpha vs SPY: {best_net_return - spy_annual_return:.2f}%")

# Business viability conclusion
print("\n" + "="*60)
print("BUSINESS VIABILITY CONCLUSION")
print("="*60)

if best_net_return > spy_annual_return:
    viability = "VIABLE"
    recommendation = "Investors would benefit from this fund vs SPY."
else:
    viability = "NOT VIABLE as standalone energy fund"
    underperformance = spy_annual_return - best_net_return
    recommendation = f"Fund underperforms SPY by {underperformance:.2f}% annually after fees."

print(f"\nViability: {viability}")
print(f"Recommendation: {recommendation}")

print("\nKey Considerations:")
print("  • Energy sector is highly volatile and cyclical")
print("  • 2002-2024 period includes major oil crashes (2008, 2014-2016, 2020)")
print("  • Tech sector dominance drove SPY outperformance")
print("  • Fees significantly erode investor returns")
print(f"  • Best strategy still trails SPY by {abs(best_net_return - spy_annual_return):.2f}% annually")

# Save assessment
assessment = {
    'spy_annual_return': spy_annual_return,
    'best_strategy': best_strategy,
    'best_gross_return': best_gross_return,
    'best_net_return_2_20_fees': best_net_return,
    'alpha_vs_spy': best_net_return - spy_annual_return,
    'viability': viability,
    'recommendation': recommendation
}

pd.DataFrame([assessment]).to_csv('../data/business_viability.csv', index=False)

# ==============================================================================
# STEP 8: GENERATE VISUALIZATIONS
# ==============================================================================

print("\n" + "="*80)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*80)

# Chart 1: Fee Impact
print("\n[1/4] Fee Impact Chart...")
fig, ax = plt.subplots(figsize=(12, 7))

fee_scenarios_plot = [
    (0, "No Fees"),
    (1, "1% Mgmt"),
    (2, "2% Mgmt"),
    (3, "3% Mgmt"),
    (4, "4% Mgmt"),
    (2, "2% + 20% Perf"),
]

net_returns_plot = []
for i, (mgmt, perf, label) in enumerate(fee_scenarios[:6]):
    net_returns_plot.append(fee_impacts[i]['net_annual_return_pct'])

labels_plot = [label for _, _, label in fee_scenarios[:6]]

bars = ax.bar(range(len(labels_plot)), net_returns_plot, color='steelblue', edgecolor='black', alpha=0.7)
ax.axhline(spy_annual_return, color='red', linestyle='--', linewidth=2, label=f'SPY Benchmark ({spy_annual_return:.1f}%)')

ax.set_ylabel('Net Annual Return (%)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Fee Structure on Investor Returns', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(labels_plot)))
ax.set_xticklabels(labels_plot, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../data/fee_impact_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: fee_impact_chart.png")

# Chart 2: Monte Carlo Distribution
print("[2/4] Monte Carlo Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(mc_results_normal['annual_return'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(mc_results_normal['annual_return'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean: {mc_results_normal['annual_return'].mean():.1f}%")
ax.set_xlabel('Annual Return (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Monte Carlo: Multivariate Normal', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
ax.hist(mc_results_bootstrap['annual_return'], bins=50, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(mc_results_bootstrap['annual_return'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean: {mc_results_bootstrap['annual_return'].mean():.1f}%")
ax.set_xlabel('Annual Return (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Monte Carlo: Bootstrap', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../data/monte_carlo_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: monte_carlo_distribution.png")

# Chart 3: Walk-Forward Results
print("[3/4] Walk-Forward Performance...")
fig, ax = plt.subplots(figsize=(14, 7))

windows = wf_results['window']
returns = wf_results['annual_return']

ax.bar(windows, returns, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax.axhline(returns.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Average: {returns.mean():.1f}%')
ax.axhline(0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Test Window', fontsize=12, fontweight='bold')
ax.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
ax.set_title('Walk-Forward Test Results (5-Year Train, 1-Year Test)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../data/walk_forward_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: walk_forward_performance.png")

# Chart 4: Alpha vs Beta
print("[4/4] Alpha vs Beta Scatter...")
fig, ax = plt.subplots(figsize=(10, 8))

for strategy in alpha_beta_df.index:
    ax.scatter(alpha_beta_df.loc[strategy, 'beta'],
               alpha_beta_df.loc[strategy, 'alpha_annual'],
               s=200, alpha=0.7, edgecolors='black', linewidth=2)
    ax.annotate(strategy.replace('_', ' '),
                xy=(alpha_beta_df.loc[strategy, 'beta'],
                    alpha_beta_df.loc[strategy, 'alpha_annual']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero Alpha')
ax.axvline(1, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Beta = 1')

ax.set_xlabel('Beta (Market Sensitivity)', fontsize=12, fontweight='bold')
ax.set_ylabel('Alpha (% p.a.)', fontsize=12, fontweight='bold')
ax.set_title('Risk-Adjusted Performance: Alpha vs Beta', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/alpha_beta_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: alpha_beta_scatter.png")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)

print("\nAll Files Generated:")
print("\n  Performance Metrics:")
print("    - full_period_metrics.csv")
print("    - alpha_beta_analysis.csv")
print("    - business_viability.csv")

print("\n  Monte Carlo:")
print("    - monte_carlo_normal_results.csv")
print("    - monte_carlo_bootstrap_results.csv")
print("    - monte_carlo_normal_summary.csv")
print("    - monte_carlo_bootstrap_summary.csv")

print("\n  Walk-Forward:")
print("    - walk_forward_results.csv")
print("    - walk_forward_summary.csv")

print("\n  Fee Analysis:")
print("    - fee_impact_analysis.csv")
print("    - fee_sensitivity_grid.csv")

print("\n  Charts:")
print("    - fee_impact_chart.png")
print("    - monte_carlo_distribution.png")
print("    - walk_forward_performance.png")
print("    - alpha_beta_scatter.png")

print("\n" + "="*80)
print("READY FOR FINAL PAPER!")
print("="*80)
