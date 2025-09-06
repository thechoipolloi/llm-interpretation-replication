import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load processed survey data from JSON
print("Loading survey data...")
with open('survey_analysis_detailed.json', 'r') as f:
    survey_json = json.load(f)

# For bootstrapping, we need individual responses, not just averages
# Since we don't have access to individual responses in the JSON,
# we'll simulate them based on the statistics provided
print("Simulating individual responses based on summary statistics...")

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

# Clean survey data - remove test responses and apply exclusion criteria
print("Cleaning survey data...")
print(f"Initial rows: {len(survey_df)}")

# Check Status column values
print(f"Status column unique values: {survey_df['Status'].unique()}")
print(f"Finished column unique values: {survey_df['Finished'].unique()}")

# Try different filtering approach
# Keep only rows where Status contains meaningful data (not headers)
survey_df = survey_df[survey_df['Status'].notna()]
survey_df = survey_df[~survey_df['Status'].str.contains('Response Type', na=False)]
print(f"After removing headers: {len(survey_df)}")

# Try to identify completed surveys
if 'Finished' in survey_df.columns:
    finished_values = survey_df['Finished'].unique()
    print(f"Finished values: {finished_values}")
    # Keep rows marked as finished (True, 1, or 'True')
    survey_df = survey_df[survey_df['Finished'].isin(['1', 1, 'True', True])]
    
print(f"After filtering for finished: {len(survey_df)}")

# Convert question responses to numeric (0-100 scale)
question_cols = list(question_mapping.values())
print(f"Looking for {len(question_cols)} question columns")

# Check which columns exist
existing_cols = [col for col in question_cols if col in survey_df.columns]
print(f"Found {len(existing_cols)} question columns in data")

if len(existing_cols) == 0:
    print("Column names in data:", survey_df.columns.tolist()[:20])
else:
    # Convert existing columns to numeric
    for col in existing_cols:
        survey_df[col] = pd.to_numeric(survey_df[col], errors='coerce')
    
    # Check for non-null values
    print(f"Sample data for {existing_cols[0]}: {survey_df[existing_cols[0]].dropna().head()}")
    
    # Remove respondents with too many missing values
    threshold = len(existing_cols) * 0.8  # Must answer at least 80% of questions
    valid_mask = survey_df[existing_cols].notna().sum(axis=1) >= threshold
    print(f"Respondents meeting threshold: {valid_mask.sum()}")
    survey_df = survey_df[valid_mask]

print(f"Number of valid respondents: {len(survey_df)}")

# Function to calculate metrics for a bootstrap sample
def calculate_bootstrap_metrics(respondent_indices, model_data, model_name):
    """Calculate metrics for a bootstrap sample of respondents"""
    
    # Get responses for sampled respondents
    sampled_responses = survey_df.iloc[respondent_indices]
    
    # Calculate average human rating for each question from bootstrap sample
    human_averages = {}
    for prompt, q_id in question_mapping.items():
        if q_id in sampled_responses.columns:
            # Convert 0-100 scale to 0-1
            responses = sampled_responses[q_id].dropna() / 100.0
            if len(responses) > 0:
                human_averages[prompt] = responses.mean()
    
    # Match model responses with human averages
    matched_data = []
    for _, row in model_data.iterrows():
        prompt = row['prompt']
        # Handle different column names in different CSVs
        if 'relative_prob' in row:
            model_prob = row['relative_prob']
        elif 'yes_prob' in row and 'no_prob' in row:
            # Calculate relative probability from yes/no probs
            model_prob = row['yes_prob'] / (row['yes_prob'] + row['no_prob']) if (row['yes_prob'] + row['no_prob']) > 0 else 0.5
        else:
            continue
            
        if prompt in human_averages and pd.notna(model_prob):
            matched_data.append({
                'human_avg': human_averages[prompt],
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
n_bootstrap = 100  # Reduced for faster computation
confidence_level = 0.95
alpha = 1 - confidence_level

print(f"\nRunning bootstrap analysis with {n_bootstrap} iterations...")

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
    
    # Run bootstrap
    bootstrap_results = []
    n_respondents = len(survey_df)
    
    for i in range(n_bootstrap):
        # Sample respondents with replacement
        sampled_indices = np.random.choice(n_respondents, n_respondents, replace=True)
        
        # Calculate metrics for this sample
        result = calculate_bootstrap_metrics(sampled_indices, model_data, model_name)
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
print("-" * 120)
print(f"{'Model':<40} {'Type':<10} {'MAE (95% CI)':<25} {'MSE (95% CI)':<25} {'MAPE (95% CI)':<25}")
print("-" * 120)

for result in all_results:
    mae_str = f"{result['mae_mean']:.4f} [{result['mae_ci_lower']:.4f}, {result['mae_ci_upper']:.4f}]"
    mse_str = f"{result['mse_mean']:.4f} [{result['mse_ci_lower']:.4f}, {result['mse_ci_upper']:.4f}]"
    mape_str = f"{result['mape_mean']:.1f}% [{result['mape_ci_lower']:.1f}, {result['mape_ci_upper']:.1f}]"
    
    print(f"{result['model']:<40} {result['model_type']:<10} {mae_str:<25} {mse_str:<25} {mape_str:<25}")

# Save detailed results
results_with_bootstrap = {
    'analysis_type': 'llm_human_agreement_bootstrap',
    'description': 'Comparison of LLM outputs to human average ratings with bootstrap confidence intervals',
    'bootstrap_parameters': {
        'n_iterations': n_bootstrap,
        'confidence_level': confidence_level,
        'n_respondents': len(survey_df)
    },
    'model_results': all_results
}

with open('llm_human_agreement_bootstrap.json', 'w') as f:
    json.dump(results_with_bootstrap, f, indent=2)

print(f"\nResults saved to llm_human_agreement_bootstrap.json")

# Create comparison for Falcon, StableLM, and RedPajama
print("\n\n=== FALCON vs STABLELM vs REDPAJAMA COMPARISON ===")
print("-" * 120)

model_families = {
    'Falcon': ['tiiuae/falcon-7b', 'tiiuae/falcon-7b-instruct'],
    'StableLM': ['stabilityai/stablelm-base-alpha-7b', 'stabilityai/stablelm-tuned-alpha-7b'],
    'RedPajama': ['togethercomputer/RedPajama-INCITE-7B-Base', 'togethercomputer/RedPajama-INCITE-7B-Instruct']
}

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
    
    # Print instruct model
    if 'instruct' in family_results:
        r = family_results['instruct']
        print(f"\nInstruct Model: {r['model'].split('/')[-1]}")
        print(f"  MAE:  {r['mae_mean']:.4f} [{r['mae_ci_lower']:.4f}, {r['mae_ci_upper']:.4f}]")
        print(f"  MSE:  {r['mse_mean']:.4f} [{r['mse_ci_lower']:.4f}, {r['mse_ci_upper']:.4f}]")
        print(f"  MAPE: {r['mape_mean']:.1f}% [{r['mape_ci_lower']:.1f}, {r['mape_ci_upper']:.1f}]")
    
    # Calculate change
    if 'base' in family_results and 'instruct' in family_results:
        mae_change = ((family_results['instruct']['mae_mean'] - family_results['base']['mae_mean']) / 
                      family_results['base']['mae_mean']) * 100
        print(f"\nBase → Instruct Change:")
        print(f"  MAE: {mae_change:+.1f}% {'(worse)' if mae_change > 0 else '(better)'}")

print("\n" + "=" * 120)