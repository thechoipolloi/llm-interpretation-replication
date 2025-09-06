import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from pathlib import Path
import os
import itertools
from scipy.stats import pearsonr, spearmanr
# Add sklearn for Cohen's kappa
from sklearn.metrics import cohen_kappa_score
from scipy import stats  # Add this for bootstrap confidence intervals

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Define the base directory for all operations
BASE_DIR = "G:/My Drive/Computational/llm_interpretation/"

# Create output directories if they don't exist
output_dir = Path(BASE_DIR) / "output"
output_dir.mkdir(exist_ok=True)
figures_dir = output_dir / "figures"
figures_dir.mkdir(exist_ok=True)

def create_model_comparison_plot(df, prompt_subset=None):
    """
    Create a plot comparing model relative probabilities against Baichuan as the reference model.
    If no Baichuan model is found, falls back to a randomly selected reference model.
    
    Args:
        df: DataFrame containing model outputs
        prompt_subset: Optional list of prompts to include (if None, use all prompts)
    """
    # Get unique prompts and models
    unique_prompts = df['prompt'].unique() if prompt_subset is None else prompt_subset
    unique_models = df['model'].unique()
    
    # Create a figure
    plt.figure(figsize=(14, 10))
    
    # Set up the plot
    ax = plt.subplot(111)
    
    # Define colors for different prompts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Store data for legend
    legend_elements = []
    
    # Use Baichuan as the reference model
    baichuan_models = [model for model in unique_models if 'baichuan' in model.lower()]
    
    if not baichuan_models:
        print("No Baichuan model found. Falling back to random selection.")
        # Randomly select a reference model that has data for all prompts
        valid_models = []
        for model in unique_models:
            model_data = df[df['model'] == model]
            if len(model_data.dropna(subset=['relative_prob'])) >= len(unique_prompts):
                valid_models.append(model)
        
        if len(valid_models) == 0:
            print("No model has data for all prompts. Using model with most data.")
            model_counts = df.groupby('model').count()['relative_prob'].sort_values(ascending=False)
            valid_models = [model_counts.index[0]]
        
        reference_model = random.choice(valid_models)
    else:
        # Use the first Baichuan model found
        reference_model = baichuan_models[0]
    
    print(f"Selected reference model: {reference_model}")
    
    # Process each model (except reference model)
    model_indices = {}  # To keep track of x-axis positions
    model_idx = 0  # Initialize counter separately from enumerate to avoid gaps
    
    for i, model in enumerate(unique_models):
        if model == reference_model:
            continue
        
        print(f"Processing model: {model}")
        model_indices[model] = model_idx
        
        # Get data for this model across all prompts
        model_data = df[df['model'] == model]
        
        # Get reference model data
        ref_model_data = df[df['model'] == reference_model]
        
        # Calculate differences for each prompt
        prompt_diffs = []
        prompt_names = []
        
        for prompt in unique_prompts:
            # Get model probability for this prompt
            prompt_model_data = model_data[model_data['prompt'] == prompt]
            if len(prompt_model_data) == 0 or prompt_model_data['relative_prob'].isna().all():
                print(f"  Skipping prompt '{prompt}' - no data for model {model}")
                continue
                
            # Get reference model probability for this prompt
            prompt_ref_data = ref_model_data[ref_model_data['prompt'] == prompt]
            if len(prompt_ref_data) == 0 or prompt_ref_data['relative_prob'].isna().all():
                print(f"  Skipping prompt '{prompt}' - no data for reference model")
                continue
            
            model_prob = prompt_model_data['relative_prob'].values[0]
            ref_prob = prompt_ref_data['relative_prob'].values[0]
            diff = model_prob - ref_prob
            
            prompt_diffs.append(diff)
            prompt_names.append(prompt)
        
        # Skip if no valid data
        if len(prompt_diffs) == 0:
            print(f"  Skipping model '{model}' - no valid data")
            continue
        
        # Add violin plot
        violin_parts = ax.violinplot([prompt_diffs], [model_idx], 
                                    widths=0.6, showmeans=False, showmedians=False, showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[model_idx % len(colors)])
            pc.set_edgecolor('none')
            pc.set_alpha(0.3)
        
        # Add jittered points for each prompt - now using the same color as the violin plot
        x_jittered = np.random.normal(model_idx, 0.08, size=len(prompt_diffs))
        for i, (x, y, prompt) in enumerate(zip(x_jittered, prompt_diffs, prompt_names)):
            # Use model color instead of prompt color
            plt.scatter(x, y, alpha=0.7, s=50, 
                       color=colors[model_idx % len(colors)])
        
        # Calculate mean and 95% confidence interval
        mean_diff = np.mean(prompt_diffs)
        if len(prompt_diffs) > 1:
            # Calculate 95% confidence interval
            lower_percentile = np.percentile(prompt_diffs, 2.5)
            upper_percentile = np.percentile(prompt_diffs, 97.5)
            
            # Add error bars for 95% CI
            plt.plot([model_idx, model_idx], [lower_percentile, upper_percentile], 
                    color='black', linewidth=2, zorder=4)
            
            # Add caps to the error bars
            cap_width = 0.1
            plt.plot([model_idx - cap_width, model_idx + cap_width], 
                    [lower_percentile, lower_percentile], color='black', linewidth=2, zorder=4)
            plt.plot([model_idx - cap_width, model_idx + cap_width], 
                    [upper_percentile, upper_percentile], color='black', linewidth=2, zorder=4)
        
        # Add mean point (larger and with black edge for visibility)
        plt.scatter(model_idx, mean_diff, color='black', 
                   s=100, zorder=5)
        
        # Add to legend elements for the model
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                       markerfacecolor=colors[model_idx % len(colors)], markersize=10, 
                                       label=f"{model.split('/')[-1]}"))
        
        # Increment model_idx after processing each non-reference model
        model_idx += 1
    
    # Add a point for the reference model (which is always 0)
    # Place it at the end of the actual models we've plotted, not based on the length of unique_models
    plt.scatter(model_idx, 0, color='black', s=100, marker='*')
    
    # Add reference model to legend elements
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='black', 
                                   markersize=10, 
                                   label=f"Reference: {reference_model.split('/')[-1]}"))
    
    # Add a horizontal line at 0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set the x-ticks without labels
    # Use model_idx instead of len(model_labels) to ensure we have the correct number of ticks
    plt.xticks(range(model_idx), [""] * model_idx, fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add labels
    plt.xlabel('Model', fontsize=20)
    plt.ylabel('Difference in Relative Probability\nfrom Reference Model', fontsize=20)
    
    # Add the legend at the bottom of the plot
    plt.legend(handles=legend_elements, fontsize=16, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Adjust layout and save with extra space at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add extra space at bottom for legend
    plt.savefig(figures_dir / 'model_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {figures_dir / 'model_comparison_plot.png'}")

def calculate_model_correlations(df, correlation_type='pearson', n_samples=1000, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate correlations between model outputs for the same prompts using bootstrapping.
    
    Args:
        df: DataFrame containing model outputs
        correlation_type: Type of correlation to calculate ('pearson' or 'spearman')
        n_samples: Number of random model pairs to sample (if None, use all pairs)
        n_bootstrap: Number of bootstrap samples to generate
        confidence_level: Confidence level for confidence intervals (default 0.95)
    
    Returns:
        Dictionary containing correlation statistics with confidence intervals
    """
    unique_prompts = df['prompt'].unique()
    unique_models = df['model'].unique()
    
    print(f"Calculating {correlation_type} correlations between models with bootstrapping...")
    print(f"Number of bootstrap samples: {n_bootstrap}")
    print(f"Confidence level: {confidence_level * 100}%")
    
    # Create a pivot table with prompts as rows and models as columns
    pivot_df = df.pivot_table(index='prompt', columns='model', values='relative_prob')
    
    # Get all model pairs (excluding self-correlations)
    model_pairs = list(itertools.combinations(unique_models, 2))
    
    # If n_samples is provided and less than the total number of pairs, sample randomly
    if n_samples is not None and n_samples < len(model_pairs):
        random.seed(42)  # For reproducibility
        model_pairs = random.sample(model_pairs, n_samples)
    
    # Store bootstrap statistics
    bootstrap_means = []
    bootstrap_medians = []
    bootstrap_stds = []
    
    # Original correlations (using all data)
    original_corr_matrix = pivot_df.corr(method=correlation_type.lower())
    original_correlations = []
    for model1, model2 in model_pairs:
        corr_value = original_corr_matrix.loc[model1, model2]
        if not np.isnan(corr_value):
            original_correlations.append(corr_value)
    
    # Calculate original statistics
    original_mean = np.mean(original_correlations)
    original_median = np.median(original_correlations)
    original_std = np.std(original_correlations)
    
    # Perform bootstrapping
    np.random.seed(42)  # For reproducibility
    n_prompts = len(unique_prompts)
    
    for i in range(n_bootstrap):
        # Sample prompts with replacement
        bootstrap_indices = np.random.choice(n_prompts, size=n_prompts, replace=True)
        bootstrap_prompts = unique_prompts[bootstrap_indices]
        
        # Create bootstrap sample of the pivot table
        bootstrap_pivot = pivot_df.loc[bootstrap_prompts]
        
        # Calculate correlation matrix for bootstrap sample
        if correlation_type.lower() == 'pearson':
            bootstrap_corr_matrix = bootstrap_pivot.corr(method='pearson')
        else:
            bootstrap_corr_matrix = bootstrap_pivot.corr(method='spearman')
        
        # Extract correlations for each pair in bootstrap sample
        bootstrap_correlations = []
        for model1, model2 in model_pairs:
            try:
                corr_value = bootstrap_corr_matrix.loc[model1, model2]
                if not np.isnan(corr_value):
                    bootstrap_correlations.append(corr_value)
            except KeyError:
                # Handle case where a model might not be in bootstrap sample
                continue
        
        # Calculate statistics for this bootstrap sample
        if bootstrap_correlations:
            bootstrap_means.append(np.mean(bootstrap_correlations))
            bootstrap_medians.append(np.median(bootstrap_correlations))
            bootstrap_stds.append(np.std(bootstrap_correlations))
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap} bootstrap samples...")
    
    # Calculate confidence intervals using percentile method
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Mean confidence interval
    mean_ci_lower = np.percentile(bootstrap_means, lower_percentile)
    mean_ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    # Median confidence interval
    median_ci_lower = np.percentile(bootstrap_medians, lower_percentile)
    median_ci_upper = np.percentile(bootstrap_medians, upper_percentile)
    
    # Standard deviation confidence interval
    std_ci_lower = np.percentile(bootstrap_stds, lower_percentile)
    std_ci_upper = np.percentile(bootstrap_stds, upper_percentile)
    
    # Calculate bootstrap standard errors
    mean_se = np.std(bootstrap_means)
    median_se = np.std(bootstrap_medians)
    std_se = np.std(bootstrap_stds)
    
    # Calculate statistics
    stats_dict = {
        'mean_correlation': original_mean,
        'mean_ci': (mean_ci_lower, mean_ci_upper),
        'mean_se': mean_se,
        'median_correlation': original_median,
        'median_ci': (median_ci_lower, median_ci_upper),
        'median_se': median_se,
        'std_correlation': original_std,
        'std_ci': (std_ci_lower, std_ci_upper),
        'std_se': std_se,
        'min_correlation': np.min(original_correlations),
        'max_correlation': np.max(original_correlations),
        'correlation_matrix': original_corr_matrix,
        'correlation_values': original_correlations,
        'bootstrap_means': bootstrap_means,
        'bootstrap_medians': bootstrap_medians,
        'bootstrap_stds': bootstrap_stds,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }
    
    return stats_dict

def get_abbreviated_model_name(model_name):
    """
    Convert full model names to abbreviated versions.
    
    Args:
        model_name: Full model name
        
    Returns:
        Abbreviated model name
    """
    original_name = model_name
    
    # Extract the last part of the model path if it contains slashes
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # Print the model name for debugging
    print(f"Processing model name: '{original_name}' -> '{model_name}'")
    
    # Define mappings for model name abbreviations
    if 'qwen' in model_name.lower():
        return 'Qwen'
    elif 'tk' in model_name.lower():
        return 'tk'
    elif 'baichuan' in model_name.lower():
        return 'Baichuan'
    elif 't0' in model_name.lower():
        return 'T0'
    elif 'bloomz' in model_name.lower():
        return 'bloomz'
    elif 'h2ogpt' in model_name.lower():
        return 'h2ogpt'
    elif 'mistral' in model_name.lower():
        print(f"  Found Mistral model: {model_name}")
        return 'Mistral'
    elif 'falcon' in model_name.lower():
        return 'falcon'
    elif 'redpajama' in model_name.lower():
        return 'RedPajama'
    else:
        # For other models, use the last part of the name and truncate if too long
        short_name = model_name.split('/')[-1]
        if len(short_name) > 10:
            return short_name[:10] + '...'
        print(f"  No specific abbreviation found for: {model_name}, using: {short_name}")
        return short_name

def plot_correlation_matrix(corr_matrix, output_path):
    """
    Create a heatmap visualization of the correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        output_path: Path to save the figure
    """
    # Print the models in the correlation matrix for debugging
    print("\nModels in correlation matrix:")
    for model in corr_matrix.index:
        print(f"  - {model}")
    
    plt.figure(figsize=(12, 10))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Create a copy of the correlation matrix with abbreviated model names
    corr_matrix_abbrev = corr_matrix.copy()
    
    # Create mapping of full names to abbreviated names
    abbreviated_names = {model: get_abbreviated_model_name(model) for model in corr_matrix.index}
    
    # Print the abbreviated names for debugging
    print("\nAbbreviated model names:")
    for original, abbreviated in abbreviated_names.items():
        print(f"  - {original} -> {abbreviated}")
    
    # Rename the index and columns with abbreviated names
    corr_matrix_abbrev.index = [abbreviated_names[model] for model in corr_matrix.index]
    corr_matrix_abbrev.columns = [abbreviated_names[model] for model in corr_matrix.columns]
    
    # Draw the heatmap with abbreviated names
    sns.heatmap(corr_matrix_abbrev, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 16})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation matrix plot saved to {output_path}")

def plot_correlation_distribution(correlations, output_path, correlation_type='pearson', 
                                 mean_ci=None, median_ci=None, std_ci=None):
    """
    Create a histogram of correlation values between model pairs with confidence intervals.
    
    Args:
        correlations: List of correlation values
        output_path: Path to save the figure
        correlation_type: Type of correlation ('pearson' or 'spearman')
        mean_ci: Tuple of (lower, upper) confidence interval for mean
        median_ci: Tuple of (lower, upper) confidence interval for median
        std_ci: Tuple of (lower, upper) confidence interval for standard deviation
    """
    plt.figure(figsize=(12, 7))
    
    # Create histogram
    sns.histplot(correlations, kde=False, bins=20, alpha=0.7)
    
    # Plot mean with confidence interval
    mean_val = np.mean(correlations)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_val:.3f}')
    
    if mean_ci is not None:
        # Add shaded region for mean CI
        plt.axvspan(mean_ci[0], mean_ci[1], alpha=0.2, color='red',
                   label=f'Mean 95% CI: [{mean_ci[0]:.3f}, {mean_ci[1]:.3f}]')
    
    # Plot median with confidence interval
    median_val = np.median(correlations)
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2,
                label=f'Median: {median_val:.3f}')
    
    if median_ci is not None:
        # Add shaded region for median CI
        plt.axvspan(median_ci[0], median_ci[1], alpha=0.2, color='green',
                   label=f'Median 95% CI: [{median_ci[0]:.3f}, {median_ci[1]:.3f}]')
    
    # Add standard deviation in the title or as text
    std_val = np.std(correlations)
    if std_ci is not None:
        plt.text(0.02, 0.98, f'Std Dev: {std_val:.3f} (95% CI: [{std_ci[0]:.3f}, {std_ci[1]:.3f}])',
                transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        plt.text(0.02, 0.98, f'Std Dev: {std_val:.3f}',
                transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Correlation Coefficient', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title(f'{correlation_type.capitalize()} Correlation Distribution with Bootstrap CIs', fontsize=20)
    plt.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation distribution plot with CIs saved to {output_path}")

def calculate_cohens_kappa(df, threshold=0.5):
    """
    Calculate Cohen's kappa between pairs of models to measure agreement beyond chance.
    
    Args:
        df: DataFrame containing model outputs
        threshold: Threshold for binarizing relative probabilities
    
    Returns:
        Dictionary containing kappa statistics
    """
    print(f"\nCalculating Cohen's kappa between model pairs...")
    
    unique_models = df['model'].unique()
    model_pairs = list(itertools.combinations(unique_models, 2))
    
    kappa_scores = []
    model_pair_names = []
    
    for model1, model2 in model_pairs:
        # Get data for both models
        data1 = df[df['model'] == model1][['prompt', 'relative_prob']]
        data2 = df[df['model'] == model2][['prompt', 'relative_prob']]
        
        # Merge on prompt to get predictions for the same prompts
        merged_data = pd.merge(data1, data2, on='prompt', suffixes=('_1', '_2'))
        
        if len(merged_data) < 2:
            continue
            
        # Binarize the predictions based on threshold
        binary_1 = (merged_data['relative_prob_1'] > threshold).astype(int)
        binary_2 = (merged_data['relative_prob_2'] > threshold).astype(int)
        
        # Calculate Cohen's kappa
        kappa = cohen_kappa_score(binary_1, binary_2)
        
        # Store the result
        kappa_scores.append(kappa)
        model_pair_names.append((model1, model2))
    
    # Calculate statistics
    kappa_stats = {
        'mean_kappa': np.mean(kappa_scores),
        'median_kappa': np.median(kappa_scores),
        'std_kappa': np.std(kappa_scores),
        'min_kappa': np.min(kappa_scores),
        'max_kappa': np.max(kappa_scores),
        'kappa_scores': kappa_scores,
        'model_pairs': model_pair_names
    }
    
    return kappa_stats

def calculate_aggregate_cohens_kappa(df, threshold=0.5, n_bootstrap=1000):
    """
    Calculate aggregate Cohen's kappa reflecting agreement across all models pooled together.
    
    Args:
        df: DataFrame containing model outputs
        threshold: Threshold for binarizing relative probabilities
        n_bootstrap: Number of bootstrap samples for calculating kappa
    
    Returns:
        Dictionary containing aggregate kappa statistics
    """
    print(f"\nCalculating aggregate Cohen's kappa across all models...")
    
    # Create a pivot table with prompts as rows and models as columns
    pivot_df = df.pivot_table(index='prompt', columns='model', values='relative_prob')
    
    # Drop prompts with missing values to ensure all models evaluated the same prompts
    complete_prompts = pivot_df.dropna().index
    if len(complete_prompts) == 0:
        print("No prompts were evaluated by all models. Using prompts with at least two model evaluations.")
        # Keep prompts with at least two model evaluations
        complete_prompts = pivot_df.dropna(thresh=2).index
    
    pivot_df = pivot_df.loc[complete_prompts]
    
    print(f"Using {len(complete_prompts)} prompts with sufficient model evaluations")
    
    # Binarize the predictions based on threshold
    binary_df = (pivot_df > threshold).astype(int)
    
    # Calculate observed agreement
    # For each prompt, count how many model pairs agree and divide by total pairs
    agreement_rates = []
    
    for prompt in binary_df.index:
        prompt_values = binary_df.loc[prompt].values
        n_models = len(prompt_values)
        if n_models < 2:
            continue
            
        # Count agreements
        agreements = 0
        total_pairs = 0
        
        for i, j in itertools.combinations(range(n_models), 2):
            if prompt_values[i] == prompt_values[j]:
                agreements += 1
            total_pairs += 1
        
        if total_pairs > 0:
            agreement_rate = agreements / total_pairs
            agreement_rates.append(agreement_rate)
    
    observed_agreement = np.mean(agreement_rates)
    
    # Calculate chance agreement properly
    # We need the overall probability of each class (0 or 1) across all models and prompts
    all_values = binary_df.values.flatten()
    p1 = np.mean(all_values)  # Probability of class 1
    p0 = 1 - p1              # Probability of class 0
    
    # Chance agreement is the probability that two random ratings would agree
    # This is p1*p1 + p0*p0 for binary classification
    chance_agreement = p1*p1 + p0*p0
    
    # Calculate Cohen's kappa
    # kappa = (observed agreement - chance agreement) / (1 - chance agreement)
    if chance_agreement < 1:
        kappa = (observed_agreement - chance_agreement) / (1 - chance_agreement)
    else:
        kappa = 0
    
    # Bootstrap for confidence intervals
    kappa_samples = []
    
    for _ in range(n_bootstrap):
        # Resample prompts with replacement
        sampled_indices = np.random.choice(len(agreement_rates), size=len(agreement_rates), replace=True)
        sampled_agreements = [agreement_rates[i] for i in sampled_indices]
        
        # For each bootstrap sample, recalculate binary proportions from the full dataset
        # by resampling with replacement
        sampled_data_indices = np.random.choice(len(all_values), size=len(all_values), replace=True)
        sampled_all_values = [all_values[i] for i in sampled_data_indices]
        
        sample_p1 = np.mean(sampled_all_values)
        sample_p0 = 1 - sample_p1
        sample_chance = sample_p1*sample_p1 + sample_p0*sample_p0
        
        sample_observed = np.mean(sampled_agreements)
        
        if sample_chance < 1:
            sample_kappa = (sample_observed - sample_chance) / (1 - sample_chance)
            kappa_samples.append(sample_kappa)
    
    if len(kappa_samples) > 0:
        kappa_ci_lower = np.percentile(kappa_samples, 2.5)
        kappa_ci_upper = np.percentile(kappa_samples, 97.5)
    else:
        kappa_ci_lower = kappa_ci_upper = np.nan
    
    # Print detailed information for debugging
    print(f"Debug information:")
    print(f"  Binary values distribution: {np.sum(all_values)} ones, {len(all_values) - np.sum(all_values)} zeros")
    print(f"  Probability of class 1: {p1:.3f}")
    print(f"  Probability of class 0: {p0:.3f}")
    print(f"  Average observed agreement rate: {observed_agreement:.3f}")
    print(f"  Calculated chance agreement rate: {chance_agreement:.3f}")
    
    # Return statistics
    kappa_stats = {
        'aggregate_kappa': kappa,
        'observed_agreement': observed_agreement,
        'chance_agreement': chance_agreement,
        'kappa_ci_lower': kappa_ci_lower,
        'kappa_ci_upper': kappa_ci_upper,
        'n_prompts': len(complete_prompts),
        'n_models': len(binary_df.columns),
        'p_class1': p1,
        'p_class0': p0
    }
    
    return kappa_stats

def plot_kappa_distribution(kappa_scores, output_path):
    """
    Create a histogram of Cohen's kappa values between model pairs.
    
    Args:
        kappa_scores: List of kappa values
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    sns.histplot(kappa_scores, kde=False, bins=20)
    plt.axvline(np.mean(kappa_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(kappa_scores):.3f}')
    plt.axvline(np.median(kappa_scores), color='green', linestyle='-', 
                label=f'Median: {np.median(kappa_scores):.3f}')
    
    # Add reference lines for kappa interpretation
    plt.axvline(0.2, color='gray', linestyle=':', alpha=0.7, 
                label='Fair agreement (0.2)')
    plt.axvline(0.4, color='gray', linestyle=':', alpha=0.7, 
                label='Moderate agreement (0.4)')
    plt.axvline(0.6, color='gray', linestyle=':', alpha=0.7, 
                label='Substantial agreement (0.6)')
    plt.axvline(0.8, color='gray', linestyle=':', alpha=0.7, 
                label='Almost perfect agreement (0.8)')
    
    plt.xlabel("Cohen's Kappa", fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Kappa distribution plot saved to {output_path}")

if __name__ == "__main__":
    # Read the CSV file
    csv_filename = os.path.join(BASE_DIR, "instruct_model_comparison_results.csv")
    print(f"Reading data from {csv_filename}")
    
    try:
        df = pd.read_csv(csv_filename)
        print(f"Loaded data with {len(df)} rows")
        
        # Display basic statistics
        print(f"Number of unique prompts: {df['prompt'].nunique()}")
        print(f"Number of unique models: {df['model'].nunique()}")
        
        # Filter out opt-iml-1.3b and Mistral models
        df = df[~df['model'].str.contains('opt-iml-1.3b')]
        df = df[~df['model'].str.contains('mistral', case=False)]
        print(f"After filtering out opt-iml-1.3b and Mistral models: {len(df)} rows, {df['model'].nunique()} models")
        
        # Create the plot
        create_model_comparison_plot(df)
        
        # Calculate model correlations
        for corr_type in ['pearson', 'spearman']:
            print(f"\nCalculating {corr_type} correlations...")
            corr_stats = calculate_model_correlations(df, correlation_type=corr_type, n_bootstrap=1000)
            
            # Print summary statistics with confidence intervals
            print(f"\n{corr_type.capitalize()} Correlation Summary Statistics (with 95% CI):")
            print(f"  Mean correlation: {corr_stats['mean_correlation']:.3f} "
                  f"(95% CI: [{corr_stats['mean_ci'][0]:.3f}, {corr_stats['mean_ci'][1]:.3f}])")
            print(f"  Median correlation: {corr_stats['median_correlation']:.3f} "
                  f"(95% CI: [{corr_stats['median_ci'][0]:.3f}, {corr_stats['median_ci'][1]:.3f}])")
            print(f"  Standard deviation: {corr_stats['std_correlation']:.3f} "
                  f"(95% CI: [{corr_stats['std_ci'][0]:.3f}, {corr_stats['std_ci'][1]:.3f}])")
            print(f"  Min correlation: {corr_stats['min_correlation']:.3f}")
            print(f"  Max correlation: {corr_stats['max_correlation']:.3f}")
            print(f"  Bootstrap standard errors - Mean: {corr_stats['mean_se']:.4f}, "
                  f"Median: {corr_stats['median_se']:.4f}, Std: {corr_stats['std_se']:.4f}")
            
            # Plot correlation matrix
            plot_correlation_matrix(
                corr_stats['correlation_matrix'], 
                figures_dir / f'model_{corr_type}_correlation_matrix.png'
            )
            
            # Plot correlation distribution
            plot_correlation_distribution(
                corr_stats['correlation_values'], 
                figures_dir / f'model_{corr_type}_correlation_distribution.png',
                correlation_type=corr_type,
                mean_ci=corr_stats['mean_ci'],
                median_ci=corr_stats['median_ci'],
                std_ci=corr_stats['std_ci']
            )
        
        # Calculate aggregate Cohen's kappa
        kappa_stats = calculate_aggregate_cohens_kappa(df)
        
        # Print kappa summary statistics
        print("\nAggregate Cohen's Kappa Statistics:")
        print(f"  Kappa: {kappa_stats['aggregate_kappa']:.3f}")
        print(f"  95% CI: [{kappa_stats['kappa_ci_lower']:.3f}, {kappa_stats['kappa_ci_upper']:.3f}]")
        print(f"  Observed agreement: {kappa_stats['observed_agreement']:.3f}")
        print(f"  Chance agreement: {kappa_stats['chance_agreement']:.3f}")
        print(f"  Number of prompts used: {kappa_stats['n_prompts']}")
        print(f"  Number of models used: {kappa_stats['n_models']}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {csv_filename}")
        print("Please run compare_instruct_models.py first to generate the results file.")
    except Exception as e:
        print(f"Error: {e}") 