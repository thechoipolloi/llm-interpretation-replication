import json
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the bootstrap results
with open('llm_human_agreement_bootstrap.json', 'r') as f:
    results = json.load(f)

# Define model families
model_families = {
    'Falcon': {
        'base': 'tiiuae/falcon-7b',
        'instruct': 'tiiuae/falcon-7b-instruct'
    },
    'StableLM': {
        'base': 'stabilityai/stablelm-base-alpha-7b',
        'instruct': 'stabilityai/stablelm-tuned-alpha-7b'
    },
    'RedPajama': {
        'base': 'togethercomputer/RedPajama-INCITE-7B-Base',
        'instruct': 'togethercomputer/RedPajama-INCITE-7B-Instruct'
    }
}

# Extract model results
model_results = {}
for result in results['model_results']:
    model_results[result['model']] = result

print("=== PER-FAMILY BASE vs INSTRUCT DIFFERENCES ===")
print("With 95% Confidence Intervals")
print("=" * 120)

# For each family, calculate differences with bootstrap CI
for family_name, family_models in model_families.items():
    print(f"\n{family_name.upper()}")
    print("-" * 80)
    
    base_model = family_models['base']
    instruct_model = family_models['instruct']
    
    if base_model not in model_results or instruct_model not in model_results:
        print(f"Missing data for {family_name}")
        continue
    
    base_data = model_results[base_model]
    instruct_data = model_results[instruct_model]
    
    # For each metric, calculate difference and bootstrap CI
    for metric in ['mae', 'mse', 'mape']:
        # Get the mean values
        base_mean = base_data[f'{metric}_mean']
        instruct_mean = instruct_data[f'{metric}_mean']
        diff_mean = instruct_mean - base_mean
        
        # For confidence interval, we need to bootstrap the difference
        # Since we have the bootstrap distributions stored as mean/std/CI,
        # we'll use a different approach: combine the uncertainties
        
        # Method 1: Using stored standard deviations (conservative approach)
        base_std = base_data[f'{metric}_std']
        instruct_std = instruct_data[f'{metric}_std']
        
        # Standard error of the difference (assuming independence)
        se_diff = np.sqrt(base_std**2 + instruct_std**2)
        
        # 95% CI for the difference
        ci_lower = diff_mean - 1.96 * se_diff
        ci_upper = diff_mean + 1.96 * se_diff
        
        # Method 2: Using the stored CIs directly (more accurate)
        # Calculate based on the CI ranges
        base_ci_range = base_data[f'{metric}_ci_upper'] - base_data[f'{metric}_ci_lower']
        instruct_ci_range = instruct_data[f'{metric}_ci_upper'] - instruct_data[f'{metric}_ci_lower']
        
        # Combined uncertainty
        combined_ci_range = np.sqrt(base_ci_range**2 + instruct_ci_range**2)
        ci_lower_v2 = diff_mean - combined_ci_range/2
        ci_upper_v2 = diff_mean + combined_ci_range/2
        
        # Calculate relative change
        relative_change = (diff_mean / base_mean) * 100
        
        # Format output based on metric type
        if metric == 'mae':
            print(f"\nMAE Difference (Instruct - Base):")
            print(f"  Base:     {base_mean:.4f} [{base_data['mae_ci_lower']:.4f}, {base_data['mae_ci_upper']:.4f}]")
            print(f"  Instruct: {instruct_mean:.4f} [{instruct_data['mae_ci_lower']:.4f}, {instruct_data['mae_ci_upper']:.4f}]")
            print(f"  Absolute Difference: {diff_mean:+.4f} [{ci_lower_v2:+.4f}, {ci_upper_v2:+.4f}]")
            print(f"  Relative Change: {relative_change:+.1f}%")
            if ci_lower_v2 * ci_upper_v2 > 0:  # Same sign = significant
                print(f"  → {'Significantly worse' if diff_mean > 0 else 'Significantly better'} (95% CI excludes 0)")
            else:
                print(f"  → Not significant (95% CI includes 0)")
                
        elif metric == 'mse':
            print(f"\nMSE Difference (Instruct - Base):")
            print(f"  Base:     {base_mean:.4f} [{base_data['mse_ci_lower']:.4f}, {base_data['mse_ci_upper']:.4f}]")
            print(f"  Instruct: {instruct_mean:.4f} [{instruct_data['mse_ci_lower']:.4f}, {instruct_data['mse_ci_upper']:.4f}]")
            print(f"  Absolute Difference: {diff_mean:+.4f} [{ci_lower_v2:+.4f}, {ci_upper_v2:+.4f}]")
            print(f"  Relative Change: {relative_change:+.1f}%")
            if ci_lower_v2 * ci_upper_v2 > 0:
                print(f"  → {'Significantly worse' if diff_mean > 0 else 'Significantly better'} (95% CI excludes 0)")
            else:
                print(f"  → Not significant (95% CI includes 0)")
                
        elif metric == 'mape':
            print(f"\nMAPE Difference (Instruct - Base):")
            print(f"  Base:     {base_mean:.1f}% [{base_data['mape_ci_lower']:.1f}, {base_data['mape_ci_upper']:.1f}]")
            print(f"  Instruct: {instruct_mean:.1f}% [{instruct_data['mape_ci_lower']:.1f}, {instruct_data['mape_ci_upper']:.1f}]")
            print(f"  Absolute Difference: {diff_mean:+.1f}% [{ci_lower_v2:+.1f}, {ci_upper_v2:+.1f}]")
            print(f"  Relative Change: {relative_change:+.1f}%")
            if ci_lower_v2 * ci_upper_v2 > 0:
                print(f"  → {'Significantly worse' if diff_mean > 0 else 'Significantly better'} (95% CI excludes 0)")
            else:
                print(f"  → Not significant (95% CI includes 0)")

# Summary table
print("\n\n=== SUMMARY TABLE ===")
print("-" * 120)
print(f"{'Family':<12} {'Metric':<6} {'Base':<12} {'Instruct':<12} {'Difference':<20} {'95% CI':<25} {'Significant?':<15}")
print("-" * 120)

for family_name, family_models in model_families.items():
    base_model = family_models['base']
    instruct_model = family_models['instruct']
    
    if base_model not in model_results or instruct_model not in model_results:
        continue
        
    base_data = model_results[base_model]
    instruct_data = model_results[instruct_model]
    
    for metric in ['mae', 'mse', 'mape']:
        base_mean = base_data[f'{metric}_mean']
        instruct_mean = instruct_data[f'{metric}_mean']
        diff_mean = instruct_mean - base_mean
        
        # Calculate CI using Method 2 from above
        base_ci_range = base_data[f'{metric}_ci_upper'] - base_data[f'{metric}_ci_lower']
        instruct_ci_range = instruct_data[f'{metric}_ci_upper'] - instruct_data[f'{metric}_ci_lower']
        combined_ci_range = np.sqrt(base_ci_range**2 + instruct_ci_range**2)
        ci_lower = diff_mean - combined_ci_range/2
        ci_upper = diff_mean + combined_ci_range/2
        
        # Determine significance
        is_significant = ci_lower * ci_upper > 0
        sig_text = "Yes" if is_significant else "No"
        
        # Format based on metric
        if metric == 'mape':
            base_str = f"{base_mean:.1f}%"
            instruct_str = f"{instruct_mean:.1f}%"
            diff_str = f"{diff_mean:+.1f}%"
            ci_str = f"[{ci_lower:+.1f}, {ci_upper:+.1f}]"
        else:
            base_str = f"{base_mean:.4f}"
            instruct_str = f"{instruct_mean:.4f}"
            diff_str = f"{diff_mean:+.4f}"
            ci_str = f"[{ci_lower:+.4f}, {ci_upper:+.4f}]"
        
        print(f"{family_name:<12} {metric.upper():<6} {base_str:<12} {instruct_str:<12} {diff_str:<20} {ci_str:<25} {sig_text:<15}")

print("\n" + "=" * 120)

# Create a more detailed analysis with bootstrapped differences
print("\n=== BOOTSTRAP-BASED DIFFERENCE ANALYSIS ===")
print("(Using Monte Carlo simulation to estimate difference distributions)")
print("-" * 80)

# For more accurate CIs, we'll simulate the differences using the stored parameters
np.random.seed(42)  # For reproducibility
n_bootstrap = 10000

for family_name, family_models in model_families.items():
    print(f"\n{family_name.upper()}")
    print("-" * 60)
    
    base_model = family_models['base']
    instruct_model = family_models['instruct']
    
    if base_model not in model_results or instruct_model not in model_results:
        continue
    
    base_data = model_results[base_model]
    instruct_data = model_results[instruct_model]
    
    for metric in ['mae', 'mse', 'mape']:
        # Simulate bootstrap distributions using normal approximation
        # This assumes the bootstrap distributions are approximately normal
        base_samples = np.random.normal(
            base_data[f'{metric}_mean'],
            base_data[f'{metric}_std'],
            n_bootstrap
        )
        instruct_samples = np.random.normal(
            instruct_data[f'{metric}_mean'],
            instruct_data[f'{metric}_std'],
            n_bootstrap
        )
        
        # Calculate differences
        diff_samples = instruct_samples - base_samples
        
        # Calculate statistics
        diff_mean = np.mean(diff_samples)
        diff_ci_lower = np.percentile(diff_samples, 2.5)
        diff_ci_upper = np.percentile(diff_samples, 97.5)
        
        # Calculate p-value (proportion of differences that don't favor base)
        if diff_mean > 0:  # Instruct is worse
            p_value = np.mean(diff_samples <= 0) * 2  # Two-tailed
        else:  # Instruct is better
            p_value = np.mean(diff_samples >= 0) * 2  # Two-tailed
        p_value = min(p_value, 1.0)
        
        # Format output
        if metric == 'mape':
            print(f"\n{metric.upper()}: {diff_mean:+.1f}% [{diff_ci_lower:+.1f}, {diff_ci_upper:+.1f}], p = {p_value:.4f}")
        else:
            print(f"\n{metric.upper()}: {diff_mean:+.4f} [{diff_ci_lower:+.4f}, {diff_ci_upper:+.4f}], p = {p_value:.4f}")
        
        if p_value < 0.05:
            direction = "worse" if diff_mean > 0 else "better"
            print(f"       → Instruct significantly {direction} than base (p < 0.05)")
        else:
            print(f"       → No significant difference (p = {p_value:.3f})")

print("\n" + "=" * 120)