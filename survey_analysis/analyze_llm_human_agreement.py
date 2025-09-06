import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available, skipping visualizations")
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Function to calculate agreement metrics between model and human averages
def calculate_agreement_metrics(model_data, model_name):
    """Calculate how well a model's outputs match human average ratings"""
    
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
                'prompt': prompt,
                'human_avg': human_averages[prompt],
                'model_prob': model_prob,
                'difference': abs(model_prob - human_averages[prompt])
            })
    
    if len(matched_data) < 10:
        return None
        
    df_matched = pd.DataFrame(matched_data)
    
    # Calculate metrics
    mae = mean_absolute_error(df_matched['human_avg'], df_matched['model_prob'])
    rmse = np.sqrt(mean_squared_error(df_matched['human_avg'], df_matched['model_prob']))
    pearson_r, pearson_p = pearsonr(df_matched['human_avg'], df_matched['model_prob'])
    spearman_r, spearman_p = spearmanr(df_matched['human_avg'], df_matched['model_prob'])
    
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((df_matched['human_avg'] - df_matched['model_prob']) / df_matched['human_avg'])) * 100
    
    # Find questions with largest disagreement
    df_matched_sorted = df_matched.sort_values('difference', ascending=False)
    worst_questions = df_matched_sorted.head(5)[['prompt', 'human_avg', 'model_prob', 'difference']].to_dict('records')
    
    return {
        'model': model_name,
        'n_questions': len(matched_data),
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'worst_questions': worst_questions,
        'matched_data': df_matched
    }

# Analyze all models
print("\n=== ANALYZING MODEL AGREEMENT WITH HUMAN AVERAGES ===\n")

all_results = []

# Analyze instruct models
for model in df_instruct['model'].unique():
    model_data = df_instruct[df_instruct['model'] == model]
    result = calculate_agreement_metrics(model_data, model)
    if result:
        result['model_type'] = 'instruct'
        all_results.append(result)

# Analyze base models if available
if has_base_models:
    for model in df_base['model'].unique():
        model_data = df_base[df_base['model'] == model]
        result = calculate_agreement_metrics(model_data, model)
        if result:
            result['model_type'] = 'base'
            all_results.append(result)

# Sort by MAE (lower is better)
all_results.sort(key=lambda x: x['mae'])

# Print results
print("Models ranked by agreement with human averages (best to worst):")
print("-" * 100)
print(f"{'Model':<40} {'Type':<10} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'Pearson r':<10} {'N':<5}")
print("-" * 100)

for result in all_results:
    print(f"{result['model']:<40} {result['model_type']:<10} {result['mae']:<8.4f} {result['rmse']:<8.4f} {result['mape']:<8.2f}% {result['pearson_r']:<10.4f} {result['n_questions']:<5}")

# Detailed analysis of best and worst models
print("\n=== DETAILED ANALYSIS ===\n")

if all_results:
    best_model = all_results[0]
    worst_model = all_results[-1]
    
    print(f"BEST MODEL: {best_model['model']}")
    print(f"  - Mean Absolute Error: {best_model['mae']:.4f}")
    print(f"  - Root Mean Square Error: {best_model['rmse']:.4f}")
    print(f"  - Correlation with human averages: {best_model['pearson_r']:.4f}")
    print(f"  - Questions with largest disagreement:")
    for q in best_model['worst_questions']:
        print(f"    '{q['prompt'][:50]}...'")
        print(f"      Human avg: {q['human_avg']:.3f}, Model: {q['model_prob']:.3f}, Diff: {q['difference']:.3f}")
    
    print(f"\nWORST MODEL: {worst_model['model']}")
    print(f"  - Mean Absolute Error: {worst_model['mae']:.4f}")
    print(f"  - Root Mean Square Error: {worst_model['rmse']:.4f}")
    print(f"  - Correlation with human averages: {worst_model['pearson_r']:.4f}")
    print(f"  - Questions with largest disagreement:")
    for q in worst_model['worst_questions']:
        print(f"    '{q['prompt'][:50]}...'")
        print(f"      Human avg: {q['human_avg']:.3f}, Model: {q['model_prob']:.3f}, Diff: {q['difference']:.3f}")

# Create visualizations
if PLOTTING_AVAILABLE:
    print("\n=== CREATING VISUALIZATIONS ===\n")

    # 1. Scatter plots for best and worst models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    if all_results:
        # Best model
        best_data = best_model['matched_data']
        ax1.scatter(best_data['human_avg'], best_data['model_prob'], alpha=0.6)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax1.set_xlabel('Human Average Rating')
        ax1.set_ylabel('Model Probability')
        ax1.set_title(f'Best Model: {best_model["model"]}\nMAE = {best_model["mae"]:.4f}, r = {best_model["pearson_r"]:.4f}')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        
        # Worst model
        worst_data = worst_model['matched_data']
        ax2.scatter(worst_data['human_avg'], worst_data['model_prob'], alpha=0.6)
        ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax2.set_xlabel('Human Average Rating')
        ax2.set_ylabel('Model Probability')
        ax2.set_title(f'Worst Model: {worst_model["model"]}\nMAE = {worst_model["mae"]:.4f}, r = {worst_model["pearson_r"]:.4f}')
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('best_worst_model_agreement.png', dpi=300)
    plt.close()

    # 2. Bar chart comparing all models by MAE
    plt.figure(figsize=(12, 8))
    model_names = [r['model'].split('/')[-1][:20] + '...' if len(r['model']) > 20 else r['model'] for r in all_results]
    maes = [r['mae'] for r in all_results]
    colors = ['blue' if r['model_type'] == 'instruct' else 'green' for r in all_results]

    bars = plt.barh(model_names, maes, color=colors)
    plt.xlabel('Mean Absolute Error (lower is better)')
    plt.title('Model Agreement with Human Average Ratings')
    plt.tight_layout()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Instruct Models'),
                       Patch(facecolor='green', label='Base Models')]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.savefig('model_mae_comparison.png', dpi=300)
    plt.close()

# 3. Question-level analysis
print("\n=== QUESTION-LEVEL ANALYSIS ===\n")

# Calculate variance across models for each question
question_variance = {}
for prompt in human_averages:
    model_probs = []
    for result in all_results:
        matching_rows = result['matched_data'][result['matched_data']['prompt'] == prompt]
        if not matching_rows.empty:
            model_probs.append(matching_rows.iloc[0]['model_prob'])
    
    if model_probs:
        question_variance[prompt] = {
            'human_avg': human_averages[prompt],
            'model_mean': np.mean(model_probs),
            'model_std': np.std(model_probs),
            'n_models': len(model_probs)
        }

# Find questions with highest disagreement among models
high_variance_questions = sorted(question_variance.items(), key=lambda x: x[1]['model_std'], reverse=True)[:10]

print("Questions with highest disagreement among models:")
for prompt, stats in high_variance_questions:
    print(f"\n'{prompt}'")
    print(f"  Human average: {stats['human_avg']:.3f}")
    print(f"  Model mean: {stats['model_mean']:.3f} ± {stats['model_std']:.3f}")

# Save results
results_summary = {
    'analysis_type': 'llm_human_agreement',
    'description': 'Comparison of LLM outputs to human average ratings per question',
    'model_results': [
        {
            'model': r['model'],
            'model_type': r['model_type'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'mape': r['mape'],
            'pearson_r': r['pearson_r'],
            'n_questions': r['n_questions']
        }
        for r in all_results
    ],
    'question_variance': question_variance
}

with open('llm_human_agreement_analysis.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n=== ANALYSIS COMPLETE ===")
print("Results saved to:")
print("  - llm_human_agreement_analysis.json")
print("  - best_worst_model_agreement.png")
print("  - model_mae_comparison.png")