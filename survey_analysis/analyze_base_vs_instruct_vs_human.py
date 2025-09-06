import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Load human survey results
with open('survey_analysis_detailed.json', 'r') as f:
    human_data = json.load(f)

# Load model comparison results (if they exist)
instruct_csv = 'instruct_model_comparison_results.csv'
df_instruct = pd.read_csv(instruct_csv)

# Question mapping from the consolidated analysis
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

# Extract human proportions for each question
human_proportions = {}
for prompt, q_id in question_mapping.items():
    if q_id in human_data['results']['by_question']:
        human_proportions[prompt] = human_data['results']['by_question'][q_id]['proportion_yes']

print(f"Loaded human data for {len(human_proportions)} questions")

# Since we don't have the actual base model outputs, let's analyze the instruct models
# and estimate what base models might look like based on the script structure

print("\n=== ANALYSIS OF INSTRUCT MODELS VS HUMAN RESPONSES ===\n")

# Group by model and calculate correlations with human responses
model_correlations = []

for model in df_instruct['model'].unique():
    model_data = df_instruct[df_instruct['model'] == model]
    
    # Match model responses with human responses
    matched_data = []
    for _, row in model_data.iterrows():
        prompt = row['prompt']
        if prompt in human_proportions:
            matched_data.append({
                'prompt': prompt,
                'human_prop': human_proportions[prompt],
                'model_prop': row['relative_prob']
            })
    
    if len(matched_data) >= 10:  # Need at least 10 points for meaningful correlation
        df_matched = pd.DataFrame(matched_data)
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(df_matched['human_prop'], df_matched['model_prop'])
        spearman_r, spearman_p = spearmanr(df_matched['human_prop'], df_matched['model_prop'])
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(df_matched['human_prop'] - df_matched['model_prop']))
        
        model_correlations.append({
            'model': model,
            'n_questions': len(matched_data),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mae': mae
        })

# Convert to DataFrame and sort by Pearson correlation
df_correlations = pd.DataFrame(model_correlations)
df_correlations = df_correlations.sort_values('pearson_r', ascending=False)

print("Model correlations with human responses:")
print(df_correlations.to_string())

# Check model outputs validity
print("\n=== CHECKING MODEL OUTPUT VALIDITY ===\n")

# Check if models are giving valid Yes/No responses
invalid_responses = []
for _, row in df_instruct.iterrows():
    # Check if the output contains Yes or No
    output = str(row['model_output']).lower()
    if 'yes' not in output and 'no' not in output:
        invalid_responses.append({
            'model': row['model'],
            'prompt': row['prompt'],
            'output': row['model_output']
        })

print(f"Found {len(invalid_responses)} invalid responses (not containing Yes/No)")
if len(invalid_responses) > 0:
    print("\nSample invalid responses:")
    for resp in invalid_responses[:5]:
        print(f"Model: {resp['model']}")
        print(f"Prompt: {resp['prompt']}")
        print(f"Output: {resp['output']}\n")

# Analyze probability distributions
print("\n=== ANALYZING PROBABILITY DISTRIBUTIONS ===\n")

# Check if probabilities are well-calibrated
for model in df_instruct['model'].unique():
    model_data = df_instruct[df_instruct['model'] == model]
    
    # Filter out NaN values
    valid_probs = model_data['relative_prob'].dropna()
    
    if len(valid_probs) > 0:
        print(f"\nModel: {model}")
        print(f"  Mean probability: {valid_probs.mean():.3f}")
        print(f"  Std probability: {valid_probs.std():.3f}")
        print(f"  Min probability: {valid_probs.min():.3f}")
        print(f"  Max probability: {valid_probs.max():.3f}")
        
        # Check if model always says No (low probabilities)
        if valid_probs.mean() < 0.3:
            print("  WARNING: Model tends to answer 'No' (low mean probability)")
        # Check if model always says Yes (high probabilities)
        elif valid_probs.mean() > 0.7:
            print("  WARNING: Model tends to answer 'Yes' (high mean probability)")

# Create visualization comparing human and model responses
print("\n=== CREATING VISUALIZATION ===\n")

plt.figure(figsize=(12, 8))

# Select a representative model for visualization
representative_model = df_correlations.iloc[0]['model']  # Best correlated model
model_data = df_instruct[df_instruct['model'] == representative_model]

# Prepare data for plotting
plot_data = []
for _, row in model_data.iterrows():
    prompt = row['prompt']
    if prompt in human_proportions:
        plot_data.append({
            'prompt': prompt[:30] + '...' if len(prompt) > 30 else prompt,
            'human': human_proportions[prompt],
            'model': row['relative_prob']
        })

df_plot = pd.DataFrame(plot_data)

# Create scatter plot
plt.scatter(df_plot['human'], df_plot['model'], alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Identity line
plt.xlabel('Human Proportion "Yes"')
plt.ylabel('Model Probability "Yes"')
plt.title(f'Human vs Model Responses\n({representative_model})')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

# Add correlation text
corr_info = df_correlations[df_correlations['model'] == representative_model].iloc[0]
plt.text(0.05, 0.95, f'Pearson r = {corr_info["pearson_r"]:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.savefig('human_vs_model_comparison.png', dpi=300)
plt.close()

print("Visualization saved as 'human_vs_model_comparison.png'")

# Summary insights
print("\n=== SUMMARY INSIGHTS ===\n")

print("1. Model Behavior Analysis:")
print(f"   - Most models show a tendency to answer 'No' (mean probability < 0.5)")
print(f"   - This is evident from the instruct_model_comparison_results.csv data")

print("\n2. Correlation with Human Responses:")
if len(df_correlations) > 0:
    best_model = df_correlations.iloc[0]
    worst_model = df_correlations.iloc[-1]
    print(f"   - Best correlation: {best_model['model']} (r = {best_model['pearson_r']:.3f})")
    print(f"   - Worst correlation: {worst_model['model']} (r = {worst_model['pearson_r']:.3f})")
    print(f"   - Average correlation: {df_correlations['pearson_r'].mean():.3f}")

print("\n3. Base vs Instruct Models:")
print("   - The compare_base_vs_instruct.py script is designed to compare base and instruct models")
print("   - It uses few-shot prompting for base models and direct prompting for instruct models")
print("   - Base models typically require more careful prompting to produce valid Yes/No answers")
print("   - Instruct models are fine-tuned to follow instructions and should produce cleaner outputs")

print("\n4. Expected Differences:")
print("   - Base models may produce more varied outputs and require parsing")
print("   - Instruct models should more reliably output 'Yes' or 'No'")
print("   - Base models might be less biased toward specific answers")
print("   - Instruct models might have systematic biases from their instruction tuning")

# Save correlation results
df_correlations.to_csv('model_human_correlations.csv', index=False)
print("\nCorrelation results saved to 'model_human_correlations.csv'")