import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load human survey results
with open('survey_analysis_detailed.json', 'r') as f:
    human_data = json.load(f)

# Load model comparison results
instruct_csv = 'instruct_model_comparison_results.csv'
df_instruct = pd.read_csv(instruct_csv)

# Also load base model results if available
try:
    df_base = pd.read_csv('model_comparison_results.csv')
    has_base_models = True
except:
    print("Base model results not found, analyzing instruct models only")
    has_base_models = False

# Question mapping
question_mapping = {
    "Is a \"screenshot\" a \"photograph\"?": "Q1_1",
    "Is \"advising\" someone \"instructing\" them?": "Q1_2",
    "Is an \"algorithm\" a \"procedure\"?": "Q1_3",
    "Is a \"drone\" an \"aircraft\"?": "Q1_4",
    "Is \"reading aloud\" a form of \"performance\"?": "Q1_5",
    "Is \"training\" an AI model \"authoring\" content?": "Q1_6",
    "Is a \"wedding\" a \"party\"?": "Q1_7",
    "Is \"streaming\" a video \"broadcasting\" that video?": "Q1_9",
    "Is \"braiding\" hair a form of \"weaving\"?": "Q1_10",
    "Is \"digging\" a form of \"construction\"?": "Q1_11",
    "Is a \"smartphone\" a \"computer\"?": "Q2_1",
    "Is a \"cactus\" a \"tree\"?": "Q2_2",
    "Is a \"bonus\" a form of \"wages\"?": "Q2_3",
    "Is \"forwarding\" an email \"sending\" that email?": "Q2_4",
    "Is a \"chatbot\" a \"service\"?": "Q2_5",
    "Is \"plagiarism\" a form of \"theft\"?": "Q2_6",
    "Is \"remote viewing\" of an event \"attending\" it?": "Q2_7",
    "Is \"whistling\" a form of \"music\"?": "Q2_9",
    "Is \"caching\" data in computer memory \"storing\" that data?": "Q2_10",
    "Is a \"waterway\" a form of \"roadway\"?": "Q2_11",
    "Is a \"deepfake\" a \"portrait\"?": "Q3_1",
    "Is \"humming\" a form of \"singing\"?": "Q3_2",
    "Is \"liking\" a social media post \"endorsing\" it?": "Q3_3",
    "Is \"herding\" animals a form of \"transporting\" them?": "Q3_4",
    "Is an \"NFT\" a \"security\"?": "Q3_5",
    "Is \"sleeping\" an \"activity\"?": "Q3_6",
    "Is a \"driverless car\" a \"motor vehicle operator\"?": "Q3_7",
    "Is a \"subscription fee\" a form of \"purchase\"?": "Q3_9",
    "Is \"mentoring\" someone a form of \"supervising\" them?": "Q3_10",
    "Is a \"biometric scan\" a form of \"signature\"?": "Q3_11",
    "Is a \"digital wallet\" a \"bank account\"?": "Q4_1",
    "Is \"dictation\" a form of \"writing\"?": "Q4_2",
    "Is a \"virtual tour\" a form of \"inspection\"?": "Q4_3",
    "Is \"bartering\" a form of \"payment\"?": "Q4_4",
    "Is \"listening\" to an audiobook \"reading\" it?": "Q4_5",
    "Is a \"nest\" a form of \"dwelling\"?": "Q4_6",
    "Is a \"QR code\" a \"document\"?": "Q4_7",
    "Is a \"tent\" a \"building\"?": "Q4_9",
    "Is a \"whisper\" a form of \"speech\"?": "Q4_10",
    "Is \"hiking\" a form of \"travel\"?": "Q4_11",
    "Is a \"recipe\" a form of \"instruction\"?": "Q5_1",
    "Is \"daydreaming\" a form of \"thinking\"?": "Q5_2",
    "Is \"gossip\" a form of \"news\"?": "Q5_3",
    "Is a \"mountain\" a form of \"hill\"?": "Q5_4",
    "Is \"walking\" a form of \"exercise\"?": "Q5_5",
    "Is a \"candle\" a \"lamp\"?": "Q5_6",
    "Is a \"trail\" a \"road\"?": "Q5_7",
    "Is \"repainting\" a house \"repairing\" it?": "Q5_9",
    "Is \"kneeling\" a form of \"sitting\"?": "Q5_10",
    "Is a \"mask\" a form of \"clothing\"?": "Q5_11"
}

# Extract human average ratings for each question
human_averages = {}
for prompt, q_id in question_mapping.items():
    if q_id in human_data['results']['by_question']:
        # The mean_response is on a 0-100 scale, convert to 0-1
        human_averages[prompt] = human_data['results']['by_question'][q_id]['mean_response'] / 100.0

print(f"Loaded human average ratings for {len(human_averages)} questions")

# Function to calculate metrics for a bootstrap sample of questions
def calculate_bootstrap_metrics(question_indices, model_data, human_avgs):
    """Calculate metrics for a bootstrap sample of questions"""
    
    # Get the questions for this bootstrap sample
    all_questions = list(human_avgs.keys())
    sampled_questions = [all_questions[i] for i in question_indices]
    
    # Match model responses with human averages for sampled questions
    matched_data = []
    for _, row in model_data.iterrows():
        prompt = row['prompt']
        if prompt in sampled_questions:
            # Handle different column names in different CSVs
            if 'relative_prob' in row:
                model_prob = row['relative_prob']
            elif 'yes_prob' in row and 'no_prob' in row:
                # Calculate relative probability from yes/no probs
                yes_prob = row['yes_prob']
                no_prob = row['no_prob']
                if (yes_prob + no_prob) > 0:
                    model_prob = yes_prob / (yes_prob + no_prob)
                else:
                    model_prob = 0.5
            else:
                continue
                
            if pd.notna(model_prob):
                matched_data.append({
                    'human_avg': human_avgs[prompt],
                    'model_prob': model_prob
                })
    
    if len(matched_data) < 10:
        return None
        
    df_matched = pd.DataFrame(matched_data)
    
    # Calculate metrics
    mae = mean_absolute_error(df_matched['human_avg'], df_matched['model_prob'])
    mse = mean_squared_error(df_matched['human_avg'], df_matched['model_prob'])
    
    # Calculate MAPE, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((df_matched['human_avg'] - df_matched['model_prob']) / df_matched['human_avg'])
        ape = ape[np.isfinite(ape)]  # Remove inf and nan values
        mape = np.mean(ape) * 100 if len(ape) > 0 else np.nan
    
    # Calculate correlation
    if len(df_matched) > 1 and df_matched['human_avg'].std() > 0 and df_matched['model_prob'].std() > 0:
        pearson_r, _ = pearsonr(df_matched['human_avg'], df_matched['model_prob'])
    else:
        pearson_r = np.nan
    
    return {
        'mae': mae,
        'mse': mse,
        'mape': mape,
        'pearson_r': pearson_r,
        'n_questions': len(matched_data)
    }

# Bootstrap parameters
n_bootstrap = 1000
confidence_level = 0.95
alpha = 1 - confidence_level

print(f"\nRunning bootstrap analysis with {n_bootstrap} iterations...")
print("Bootstrap sampling questions (not respondents)...")

# Analyze all models with bootstrapping
all_results = []

# Get all unique models
all_models = []
if has_base_models:
    all_models.extend([(model, 'base', df_base) for model in df_base['model'].unique()])
all_models.extend([(model, 'instruct', df_instruct) for model in df_instruct['model'].unique()])

# Process each model
for model_name, model_type, df_source in all_models:
    print(f"Processing {model_name}...")
    
    model_data = df_source[df_source['model'] == model_name]
    
    # Run bootstrap on questions
    bootstrap_results = []
    n_questions = len(human_averages)
    
    for i in range(n_bootstrap):
        # Sample questions with replacement
        sampled_indices = np.random.choice(n_questions, n_questions, replace=True)
        
        # Calculate metrics for this sample
        result = calculate_bootstrap_metrics(sampled_indices, model_data, human_averages)
        if result:
            bootstrap_results.append(result)
    
    if len(bootstrap_results) < 100:  # Need at least 100 successful bootstraps
        print(f"  Skipping {model_name} - insufficient bootstrap samples")
        continue
    
    # Calculate statistics
    metrics = {
        'model': model_name,
        'model_type': model_type,
        'n_bootstrap': len(bootstrap_results)
    }
    
    # For each metric, calculate mean and confidence intervals
    for metric in ['mae', 'mse', 'mape', 'pearson_r']:
        values = [r[metric] for r in bootstrap_results if not np.isnan(r[metric])]
        if len(values) > 0:
            metrics[f'{metric}_mean'] = np.mean(values)
            metrics[f'{metric}_ci_lower'] = np.percentile(values, (alpha/2) * 100)
            metrics[f'{metric}_ci_upper'] = np.percentile(values, (1 - alpha/2) * 100)
            metrics[f'{metric}_std'] = np.std(values)
        else:
            metrics[f'{metric}_mean'] = np.nan
            metrics[f'{metric}_ci_lower'] = np.nan
            metrics[f'{metric}_ci_upper'] = np.nan
            metrics[f'{metric}_std'] = np.nan
    
    all_results.append(metrics)

# Sort by MAE
all_results.sort(key=lambda x: x['mae_mean'])

# Print results
print("\n=== BOOTSTRAP ANALYSIS RESULTS ===")
print(f"Models ranked by agreement with human averages (best to worst)")
print(f"Confidence intervals: {confidence_level*100:.0f}%")
print(f"Note: Bootstrap performed on questions (not respondents)")
print("-" * 120)
print(f"{'Model':<40} {'Type':<10} {'MAE (95% CI)':<25} {'MSE (95% CI)':<25} {'MAPE (95% CI)':<25}")
print("-" * 120)

for result in all_results:
    mae_str = f"{result['mae_mean']:.4f} [{result['mae_ci_lower']:.4f}, {result['mae_ci_upper']:.4f}]"
    mse_str = f"{result['mse_mean']:.4f} [{result['mse_ci_lower']:.4f}, {result['mse_ci_upper']:.4f}]"
    mape_str = f"{result['mape_mean']:.1f}% [{result['mape_ci_lower']:.1f}, {result['mape_ci_upper']:.1f}]"
    
    print(f"{result['model']:<40} {result['model_type']:<10} {mae_str:<25} {mse_str:<25} {mape_str:<25}")

# Focus on Falcon, StableLM, and RedPajama
print("\n\n=== FALCON vs STABLELM vs REDPAJAMA COMPARISON ===")
print("-" * 120)

model_families = {
    'Falcon': ['tiiuae/falcon-7b', 'tiiuae/falcon-7b-instruct'],
    'StableLM': ['stabilityai/stablelm-base-alpha-7b', 'stabilityai/stablelm-tuned-alpha-7b'],
    'RedPajama': ['togethercomputer/RedPajama-INCITE-7B-Base', 'togethercomputer/RedPajama-INCITE-7B-Instruct']
}

comparison_summary = []

for family, models in model_families.items():
    print(f"\n{family.upper()}")
    print("-" * 80)
    
    family_results = {}
    for result in all_results:
        if result['model'] in models:
            if 'instruct' in result['model'].lower() or 'tuned' in result['model'].lower():
                model_type = 'instruct'
            else:
                model_type = 'base'
            family_results[model_type] = result
    
    # Print base model
    if 'base' in family_results:
        r = family_results['base']
        print(f"Base Model: {r['model'].split('/')[-1]}")
        print(f"  MAE:  {r['mae_mean']:.4f} [{r['mae_ci_lower']:.4f}, {r['mae_ci_upper']:.4f}]")
        print(f"  MSE:  {r['mse_mean']:.4f} [{r['mse_ci_lower']:.4f}, {r['mse_ci_upper']:.4f}]")
        print(f"  MAPE: {r['mape_mean']:.1f}% [{r['mape_ci_lower']:.1f}, {r['mape_ci_upper']:.1f}]")
        
        comparison_summary.append({
            'Family': family,
            'Type': 'Base',
            'MAE': r['mae_mean'],
            'MSE': r['mse_mean'],
            'MAPE': r['mape_mean']
        })
    
    # Print instruct model
    if 'instruct' in family_results:
        r = family_results['instruct']
        print(f"\nInstruct Model: {r['model'].split('/')[-1]}")
        print(f"  MAE:  {r['mae_mean']:.4f} [{r['mae_ci_lower']:.4f}, {r['mae_ci_upper']:.4f}]")
        print(f"  MSE:  {r['mse_mean']:.4f} [{r['mse_ci_lower']:.4f}, {r['mse_ci_upper']:.4f}]")
        print(f"  MAPE: {r['mape_mean']:.1f}% [{r['mape_ci_lower']:.1f}, {r['mape_ci_upper']:.1f}]")
        
        comparison_summary.append({
            'Family': family,
            'Type': 'Instruct',
            'MAE': r['mae_mean'],
            'MSE': r['mse_mean'],
            'MAPE': r['mape_mean']
        })
    
    # Calculate change
    if 'base' in family_results and 'instruct' in family_results:
        mae_change = ((family_results['instruct']['mae_mean'] - family_results['base']['mae_mean']) / 
                      family_results['base']['mae_mean']) * 100
        print(f"\nBase → Instruct Change:")
        print(f"  MAE: {mae_change:+.1f}% {'(worse)' if mae_change > 0 else '(better)'}")

# Create summary DataFrame
print("\n\n=== SUMMARY TABLE ===")
df_summary = pd.DataFrame(comparison_summary)
df_summary = df_summary.sort_values('MAE')
print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# Calculate overall metrics for base vs instruct models
print("\n\n=== OVERALL BASE vs INSTRUCT COMPARISON ===")
print("-" * 120)

# Separate base and instruct models
base_models = [r for r in all_results if r['model_type'] == 'base']
instruct_models = [r for r in all_results if r['model_type'] == 'instruct']

# Function to calculate bootstrap confidence interval and p-value for difference
def calculate_difference_stats(group1_values, group2_values, n_bootstrap=10000):
    """Calculate difference statistics with bootstrap CI and p-value"""
    # Calculate observed difference
    mean1 = np.mean(group1_values)
    mean2 = np.mean(group2_values)
    observed_diff = mean1 - mean2
    
    # Bootstrap for CI
    bootstrap_diffs = []
    n1, n2 = len(group1_values), len(group2_values)
    
    for _ in range(n_bootstrap):
        # Resample from each group
        sample1 = np.random.choice(group1_values, n1, replace=True)
        sample2 = np.random.choice(group2_values, n2, replace=True)
        bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
    
    # Calculate CI
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    # Calculate p-value (two-sided test)
    # Null hypothesis: no difference between groups
    pooled = np.concatenate([group1_values, group2_values])
    null_diffs = []
    
    for _ in range(n_bootstrap):
        # Permutation test: randomly assign to groups
        shuffled = np.random.permutation(pooled)
        null_diff = np.mean(shuffled[:n1]) - np.mean(shuffled[n1:])
        null_diffs.append(null_diff)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    
    return observed_diff, ci_lower, ci_upper, p_value

# Calculate overall statistics for each metric
for metric in ['mae', 'mse', 'mape']:
    print(f"\n{metric.upper()}:")
    
    # Get values for base models
    base_values = [r[f'{metric}_mean'] for r in base_models if not np.isnan(r[f'{metric}_mean'])]
    # Get values for instruct models  
    instruct_values = [r[f'{metric}_mean'] for r in instruct_models if not np.isnan(r[f'{metric}_mean'])]
    
    # Calculate means and CIs
    base_mean = np.mean(base_values)
    base_ci_lower = np.percentile(base_values, 2.5)
    base_ci_upper = np.percentile(base_values, 97.5)
    
    instruct_mean = np.mean(instruct_values)
    instruct_ci_lower = np.percentile(instruct_values, 2.5)
    instruct_ci_upper = np.percentile(instruct_values, 97.5)
    
    # Format based on metric type
    if metric == 'mape':
        print(f"  Base models:     {base_mean:.1f}% [{base_ci_lower:.1f}, {base_ci_upper:.1f}]")
        print(f"  Instruct models: {instruct_mean:.1f}% [{instruct_ci_lower:.1f}, {instruct_ci_upper:.1f}]")
    else:
        print(f"  Base models:     {base_mean:.4f} [{base_ci_lower:.4f}, {base_ci_upper:.4f}]")
        print(f"  Instruct models: {instruct_mean:.4f} [{instruct_ci_lower:.4f}, {instruct_ci_upper:.4f}]")
    
    # Calculate difference statistics
    diff, diff_ci_lower, diff_ci_upper, p_value = calculate_difference_stats(base_values, instruct_values)
    
    if metric == 'mape':
        print(f"  Difference (Base - Instruct): {diff:.1f}% [{diff_ci_lower:.1f}, {diff_ci_upper:.1f}], p = {p_value:.4f}")
    else:
        print(f"  Difference (Base - Instruct): {diff:.4f} [{diff_ci_lower:.4f}, {diff_ci_upper:.4f}], p = {p_value:.4f}")
    
    if p_value < 0.05:
        if diff < 0:
            print(f"  → Instruct models significantly worse (p < 0.05)")
        else:
            print(f"  → Base models significantly worse (p < 0.05)")
    else:
        print(f"  → No significant difference (p ≥ 0.05)")

# Additional analysis: matched pairs for models that have both base and instruct versions
print("\n\n=== MATCHED PAIRS ANALYSIS ===")
print("(For model families with both base and instruct versions)")
print("-" * 120)

# Find matched pairs
matched_pairs = []
for family, models in model_families.items():
    base_result = None
    instruct_result = None
    
    for result in all_results:
        if result['model'] in models:
            if 'instruct' in result['model'].lower() or 'tuned' in result['model'].lower():
                instruct_result = result
            else:
                base_result = result
    
    if base_result and instruct_result:
        matched_pairs.append({
            'family': family,
            'base': base_result,
            'instruct': instruct_result
        })

# Calculate paired differences
for metric in ['mae', 'mse', 'mape']:
    print(f"\n{metric.upper()} - Paired Differences:")
    
    differences = []
    for pair in matched_pairs:
        diff = pair['instruct'][f'{metric}_mean'] - pair['base'][f'{metric}_mean']
        differences.append(diff)
        
        if metric == 'mape':
            print(f"  {pair['family']}: {diff:+.1f}%")
        else:
            print(f"  {pair['family']}: {diff:+.4f}")
    
    # Calculate mean difference and CI
    mean_diff = np.mean(differences)
    se_diff = np.std(differences) / np.sqrt(len(differences))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    # Paired t-test p-value (one-sample t-test on differences)
    t_stat = mean_diff / se_diff if se_diff > 0 else 0
    # Approximate p-value using normal distribution
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    if metric == 'mape':
        print(f"\n  Mean paired difference: {mean_diff:+.1f}% [{ci_lower:+.1f}, {ci_upper:+.1f}], p = {p_value:.4f}")
    else:
        print(f"\n  Mean paired difference: {mean_diff:+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}], p = {p_value:.4f}")

# Save results
results_with_bootstrap = {
    'analysis_type': 'llm_human_agreement_bootstrap_questions',
    'description': 'Comparison of LLM outputs to human average ratings with bootstrap confidence intervals (sampling questions)',
    'bootstrap_parameters': {
        'n_iterations': n_bootstrap,
        'confidence_level': confidence_level,
        'bootstrap_method': 'questions_with_replacement'
    },
    'model_results': all_results,
    'overall_comparison': {
        'base_models_count': len(base_models),
        'instruct_models_count': len(instruct_models),
        'metrics': {}
    }
}

# Add overall metrics to results
for metric in ['mae', 'mse', 'mape']:
    base_values = [r[f'{metric}_mean'] for r in base_models if not np.isnan(r[f'{metric}_mean'])]
    instruct_values = [r[f'{metric}_mean'] for r in instruct_models if not np.isnan(r[f'{metric}_mean'])]
    diff, diff_ci_lower, diff_ci_upper, p_value = calculate_difference_stats(base_values, instruct_values)
    
    results_with_bootstrap['overall_comparison']['metrics'][metric] = {
        'base_mean': np.mean(base_values),
        'base_ci': [np.percentile(base_values, 2.5), np.percentile(base_values, 97.5)],
        'instruct_mean': np.mean(instruct_values),
        'instruct_ci': [np.percentile(instruct_values, 2.5), np.percentile(instruct_values, 97.5)],
        'difference': diff,
        'difference_ci': [diff_ci_lower, diff_ci_upper],
        'p_value': p_value
    }

with open('llm_human_agreement_bootstrap.json', 'w') as f:
    json.dump(results_with_bootstrap, f, indent=2)

print(f"\n\nResults saved to llm_human_agreement_bootstrap.json")
print("\n" + "=" * 120)