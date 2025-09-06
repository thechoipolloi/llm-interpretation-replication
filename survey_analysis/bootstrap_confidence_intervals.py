import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import json
from collections import defaultdict
from tqdm import tqdm

# Load consolidated analysis results
with open('consolidated_analysis_results.json', 'r') as f:
    consolidated = json.load(f)

# Load detailed human data
with open('survey_analysis_detailed.json', 'r') as f:
    detailed = json.load(f)

# Load model data
df_models = pd.read_csv('model_comparison_results.csv')

# Get the question mapping
question_mapping = consolidated['matching_stats']['matches']

# Process model responses
model_responses = defaultdict(dict)
for _, row in df_models.iterrows():
    model = row['model']
    prompt = row['prompt']
    
    try:
        yes_prob = float(row['yes_prob'])
        no_prob = float(row['no_prob'])
        if yes_prob + no_prob > 0:
            rel_prob = yes_prob / (yes_prob + no_prob)
        else:
            rel_prob = float('nan')
    except:
        rel_prob = float('nan')
    
    model_responses[model][prompt] = {
        'type': row['base_or_instruct'],
        'rel_prob': rel_prob
    }

print(f"Loaded {len(model_responses)} models")

# Group assignments for the survey structure
group_assignments = {
    1: ['Q1_' + str(i) for i in range(1, 12) if i != 8],
    2: ['Q2_' + str(i) for i in range(1, 12) if i != 8],
    3: ['Q3_' + str(i) for i in range(1, 12) if i != 8],
    4: ['Q4_' + str(i) for i in range(1, 12) if i != 8],
    5: ['Q5_' + str(i) for i in range(1, 12) if i != 8]
}

def calculate_individual_correlations(model_responses, n_samples=100, seed=None):
    """Calculate correlations between models and simulated individuals"""
    if seed is not None:
        np.random.seed(seed)
    
    model_individual_corrs = defaultdict(list)
    
    for model, responses in model_responses.items():
        model_type = responses[list(responses.keys())[0]]['type']
        
        # Sample random human-model pairs
        for _ in range(n_samples):
            # Pick a random group
            group = np.random.randint(1, 6)
            group_questions = group_assignments[group]
            
            # Create synthetic human based on group statistics
            human_vals = []
            model_vals = []
            
            for q_id in group_questions:
                # Find corresponding prompt
                prompt = None
                for p, q in question_mapping.items():
                    if q == q_id:
                        prompt = p
                        break
                
                if prompt and prompt in responses and q_id in detailed['results']['by_question']:
                    # Simulate individual variation
                    mean = detailed['results']['by_question'][q_id]['mean_response'] / 100.0
                    std = detailed['results']['by_question'][q_id]['std_response'] / 100.0
                    human_val = np.clip(np.random.normal(mean, std), 0, 1)
                    
                    human_vals.append(human_val)
                    model_vals.append(responses[prompt]['rel_prob'])
            
            if len(human_vals) >= 8 and not any(np.isnan(model_vals)):
                try:
                    corr, _ = pearsonr(human_vals, model_vals)
                    if not np.isnan(corr):
                        model_individual_corrs[model].append(corr)
                except:
                    pass
    
    return model_individual_corrs

# Calculate base correlations
print("\nCalculating base correlations...")
base_corrs = calculate_individual_correlations(model_responses, n_samples=500, seed=42)

# Calculate statistics for each model
model_stats = {}
for model, corrs in base_corrs.items():
    if corrs:
        model_type = model_responses[model][list(model_responses[model].keys())[0]]['type']
        model_stats[model] = {
            'type': model_type,
            'mean_corr': np.mean(corrs),
            'std_corr': np.std(corrs),
            'n_correlations': len(corrs),
            'correlations': corrs
        }

# Bootstrap for confidence intervals
print("\nPerforming bootstrap analysis (this may take a moment)...")
n_bootstrap = 10000
bootstrap_results = {
    'base': {'means': [], 'all_corrs': []},
    'instruct': {'means': [], 'all_corrs': []}
}

for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
    # Resample with different random seed
    boot_corrs = calculate_individual_correlations(model_responses, n_samples=100, seed=i)
    
    # Separate by type and calculate means
    base_corrs_boot = []
    instruct_corrs_boot = []
    
    for model, corrs in boot_corrs.items():
        if corrs:
            model_type = model_responses[model][list(model_responses[model].keys())[0]]['type']
            if model_type == 'base':
                base_corrs_boot.extend(corrs)
            else:
                instruct_corrs_boot.extend(corrs)
    
    if base_corrs_boot:
        bootstrap_results['base']['means'].append(np.mean(base_corrs_boot))
        bootstrap_results['base']['all_corrs'].append(base_corrs_boot)
    
    if instruct_corrs_boot:
        bootstrap_results['instruct']['means'].append(np.mean(instruct_corrs_boot))
        bootstrap_results['instruct']['all_corrs'].append(instruct_corrs_boot)

# Calculate confidence intervals
def calculate_ci(data, confidence=0.95):
    """Calculate confidence interval using percentile method"""
    alpha = 1 - confidence
    lower = np.percentile(data, (alpha/2) * 100)
    upper = np.percentile(data, (1 - alpha/2) * 100)
    return lower, upper

print("\n=== BOOTSTRAP RESULTS (95% Confidence Intervals) ===\n")

# Overall statistics
base_all = []
instruct_all = []
for model, stats in model_stats.items():
    if stats['type'] == 'base':
        base_all.extend(stats['correlations'])
    else:
        instruct_all.extend(stats['correlations'])

base_mean = np.mean(base_all)
instruct_mean = np.mean(instruct_all)
base_ci = calculate_ci(bootstrap_results['base']['means'])
instruct_ci = calculate_ci(bootstrap_results['instruct']['means'])

print(f"Base models:")
print(f"  Mean correlation: {base_mean:.4f}")
print(f"  95% CI: [{base_ci[0]:.4f}, {base_ci[1]:.4f}]")
print(f"  CI width: {base_ci[1] - base_ci[0]:.4f}")

print(f"\nInstruct models:")
print(f"  Mean correlation: {instruct_mean:.4f}")
print(f"  95% CI: [{instruct_ci[0]:.4f}, {instruct_ci[1]:.4f}]")
print(f"  CI width: {instruct_ci[1] - instruct_ci[0]:.4f}")

# Difference between base and instruct
diff_means = []
for i in range(n_bootstrap):
    if i < len(bootstrap_results['base']['means']) and i < len(bootstrap_results['instruct']['means']):
        diff_means.append(bootstrap_results['base']['means'][i] - bootstrap_results['instruct']['means'][i])

diff_mean = base_mean - instruct_mean
diff_ci = calculate_ci(diff_means)

print(f"\nDifference (Base - Instruct):")
print(f"  Mean difference: {diff_mean:.4f}")
print(f"  95% CI: [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")

if diff_ci[0] > 0:
    print("  ✓ Base models are significantly better (CI excludes 0)")
elif diff_ci[1] < 0:
    print("  ✓ Instruct models are significantly better (CI excludes 0)")
else:
    print("  ✗ No significant difference (CI includes 0)")

# Per-model confidence intervals
print("\n\n=== PER-MODEL CONFIDENCE INTERVALS ===\n")

# Bootstrap individual models
model_cis = {}
print("Bootstrapping individual models...")

for model in tqdm(model_stats.keys(), desc="Models"):
    boot_means = []
    
    # Resample from the model's correlations
    orig_corrs = model_stats[model]['correlations']
    
    for _ in range(1000):  # Fewer iterations per model
        if len(orig_corrs) > 0:
            # Resample with replacement
            resampled = np.random.choice(orig_corrs, size=len(orig_corrs), replace=True)
            boot_means.append(np.mean(resampled))
    
    if boot_means:
        ci = calculate_ci(boot_means)
        model_cis[model] = {
            'mean': model_stats[model]['mean_corr'],
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'type': model_stats[model]['type']
        }

# Sort by mean correlation
sorted_models = sorted(model_cis.items(), key=lambda x: x[1]['mean'], reverse=True)

print(f"\n{'Model':<50} {'Type':<10} {'Mean':<8} {'95% CI':<20}")
print("-" * 90)
for model, stats in sorted_models:
    ci_str = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
    print(f"{model:<50} {stats['type']:<10} {stats['mean']:>7.3f} {ci_str:<20}")

# Family comparisons with CIs
print("\n\n=== FAMILY COMPARISONS WITH CONFIDENCE INTERVALS ===\n")

families = {
    't5': {'base': 'google/t5-v1_1-base', 'instruct': 'google/flan-t5-base'},
    'falcon': {'base': 'tiiuae/falcon-7b', 'instruct': 'tiiuae/falcon-7b-instruct'},
    'bloom': {'base': 'bigscience/bloom-7b1', 'instruct': 'bigscience/bloomz-7b1'},
    'stablelm': {'base': 'stabilityai/stablelm-base-alpha-7b', 'instruct': 'stabilityai/stablelm-tuned-alpha-7b'},
    'redpajama': {'base': 'togethercomputer/RedPajama-INCITE-7B-Base', 'instruct': 'togethercomputer/RedPajama-INCITE-7B-Instruct'},
    'pythia': {'base': 'EleutherAI/pythia-6.9b', 'instruct': 'databricks/dolly-v2-7b'}
}

print(f"{'Family':<15} {'Base':<20} {'Instruct':<20} {'Difference':<15}")
print("-" * 70)

for family, models in families.items():
    if models['base'] in model_cis and models['instruct'] in model_cis:
        base_stats = model_cis[models['base']]
        inst_stats = model_cis[models['instruct']]
        
        base_str = f"{base_stats['mean']:.3f} [{base_stats['ci_lower']:.3f}, {base_stats['ci_upper']:.3f}]"
        inst_str = f"{inst_stats['mean']:.3f} [{inst_stats['ci_lower']:.3f}, {inst_stats['ci_upper']:.3f}]"
        diff = base_stats['mean'] - inst_stats['mean']
        
        # Check if CIs overlap
        overlap = not (base_stats['ci_upper'] < inst_stats['ci_lower'] or 
                      inst_stats['ci_upper'] < base_stats['ci_lower'])
        
        sig_marker = "" if overlap else "*"
        diff_str = f"{diff:>6.3f}{sig_marker}"
        
        print(f"{family:<15} {base_str:<20} {inst_str:<20} {diff_str:<15}")

print("\n* = Non-overlapping confidence intervals (suggesting significant difference)")

# Save results
results = {
    'methodology': 'Bootstrap confidence intervals for individual human-model correlations',
    'n_bootstrap': n_bootstrap,
    'overall_results': {
        'base': {
            'mean': float(base_mean),
            'ci_lower': float(base_ci[0]),
            'ci_upper': float(base_ci[1])
        },
        'instruct': {
            'mean': float(instruct_mean),
            'ci_lower': float(instruct_ci[0]),
            'ci_upper': float(instruct_ci[1])
        },
        'difference': {
            'mean': float(diff_mean),
            'ci_lower': float(diff_ci[0]),
            'ci_upper': float(diff_ci[1]),
            'significant': bool(diff_ci[0] > 0 or diff_ci[1] < 0)
        }
    },
    'per_model_results': {
        model: {
            'type': stats['type'],
            'mean': float(stats['mean']),
            'ci_lower': float(stats['ci_lower']),
            'ci_upper': float(stats['ci_upper'])
        }
        for model, stats in model_cis.items()
    }
}

with open('bootstrap_confidence_intervals.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nResults saved to 'bootstrap_confidence_intervals.json'")