import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ks_2samp
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_llm_data():
    """Load LLM model comparison results"""
    df_instruct = pd.read_csv('instruct_model_comparison_results.csv')
    df_base = pd.read_csv('model_comparison_results.csv')
    return df_instruct, df_base

def load_human_data():
    """Load human survey results"""
    with open('survey_analysis_detailed.json', 'r') as f:
        human_data = json.load(f)
    
    df = pd.read_csv('word_meaning_survey_results.csv')
    df = df[2:].reset_index(drop=True)
    
    # Convert Duration to numeric
    df['Duration (in seconds)'] = pd.to_numeric(df['Duration (in seconds)'], errors='coerce')
    
    # Get question columns
    question_cols = []
    for group in range(1, 6):
        for question in range(1, 12):
            col = f'Q{group}_{question}'
            if col in df.columns:
                question_cols.append(col)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, question_cols, human_data

def calculate_llm_correlations_with_pvalues(df_instruct, df_base):
    """Calculate pairwise correlations between LLM models with p-values"""
    
    # Combine base and instruct models
    df_combined = pd.concat([df_base, df_instruct], ignore_index=True)
    
    # Get unique models
    models = df_combined['model'].unique()
    
    # Prepare data matrix: rows = questions, columns = models
    questions = df_combined['prompt'].unique()
    
    # Create matrix
    model_responses = {}
    for model in models:
        model_data = df_combined[df_combined['model'] == model]
        responses = {}
        for _, row in model_data.iterrows():
            prompt = row['prompt']
            if 'relative_prob' in row:
                prob = row['relative_prob']
            elif 'yes_prob' in row and 'no_prob' in row:
                total = row['yes_prob'] + row['no_prob']
                prob = row['yes_prob'] / total if total > 0 else 0.5
            else:
                continue
            responses[prompt] = prob
        model_responses[model] = responses
    
    # Calculate pairwise correlations with p-values
    correlations = []
    for model1, model2 in combinations(models, 2):
        # Get common questions
        common_questions = set(model_responses[model1].keys()) & set(model_responses[model2].keys())
        
        if len(common_questions) > 10:
            values1 = [model_responses[model1][q] for q in common_questions]
            values2 = [model_responses[model2][q] for q in common_questions]
            
            # Remove NaN pairs
            pairs = [(v1, v2) for v1, v2 in zip(values1, values2) 
                     if not (pd.isna(v1) or pd.isna(v2))]
            
            if len(pairs) > 10:
                v1, v2 = zip(*pairs)
                corr, p_value = pearsonr(v1, v2)
                
                correlations.append({
                    'model1': model1,
                    'model2': model2,
                    'correlation': corr,
                    'p_value': p_value,
                    'n_questions': len(pairs),
                    'significant': p_value < 0.05
                })
    
    return correlations

def calculate_human_correlations_with_pvalues(df, question_cols):
    """Calculate pairwise correlations between human raters with p-values"""
    
    correlations = []
    
    # Process by question group
    for group in range(1, 6):
        # Get questions for this group (excluding attention check)
        group_questions = [f'Q{group}_{i}' for i in range(1, 12) if i != 8]
        
        # Get respondents who answered this group
        group_data = df[df[f'Q{group}_1'].notna()].copy()
        
        if len(group_data) > 1:
            # Calculate pairwise correlations between respondents
            for i in range(len(group_data)):
                for j in range(i+1, len(group_data)):
                    rater1_values = []
                    rater2_values = []
                    
                    for question in group_questions:
                        if pd.notna(group_data.iloc[i][question]) and pd.notna(group_data.iloc[j][question]):
                            rater1_values.append(group_data.iloc[i][question])
                            rater2_values.append(group_data.iloc[j][question])
                    
                    # Need at least 3 questions in common for meaningful correlation
                    if len(rater1_values) >= 3:
                        corr, p_value = pearsonr(rater1_values, rater2_values)
                        
                        if not np.isnan(corr):
                            correlations.append({
                                'group': group,
                                'rater1_idx': i,
                                'rater2_idx': j,
                                'correlation': corr,
                                'p_value': p_value,
                                'n_questions': len(rater1_values),
                                'significant': p_value < 0.05
                            })
    
    return correlations

def compare_distributions(llm_correlations, human_correlations):
    """Compare LLM and human correlation distributions"""
    
    llm_corr_values = [c['correlation'] for c in llm_correlations if not np.isnan(c['correlation'])]
    human_corr_values = [c['correlation'] for c in human_correlations if not np.isnan(c['correlation'])]
    
    # Mann-Whitney U test (non-parametric)
    mw_statistic, mw_pvalue = mannwhitneyu(llm_corr_values, human_corr_values, alternative='two-sided')
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = ks_2samp(llm_corr_values, human_corr_values)
    
    # T-test (assumes normality)
    t_statistic, t_pvalue = stats.ttest_ind(llm_corr_values, human_corr_values)
    
    # Calculate effect size (Cohen's d)
    llm_mean = np.mean(llm_corr_values)
    llm_std = np.std(llm_corr_values)
    human_mean = np.mean(human_corr_values)
    human_std = np.std(human_corr_values)
    
    pooled_std = np.sqrt((llm_std**2 + human_std**2) / 2)
    cohens_d = (llm_mean - human_mean) / pooled_std
    
    # Filter out NaN correlations for significant count
    valid_llm_correlations = [c for c in llm_correlations if not np.isnan(c['correlation'])]
    valid_human_correlations = [c for c in human_correlations if not np.isnan(c['correlation'])]
    
    return {
        'llm_stats': {
            'mean': llm_mean,
            'std': llm_std,
            'median': np.median(llm_corr_values),
            'n_pairs': len(llm_corr_values),
            'significant_pairs': sum(1 for c in valid_llm_correlations if c['significant']),
            'proportion_significant': sum(1 for c in valid_llm_correlations if c['significant']) / len(valid_llm_correlations) if valid_llm_correlations else 0
        },
        'human_stats': {
            'mean': human_mean,
            'std': human_std,
            'median': np.median(human_corr_values),
            'n_pairs': len(human_corr_values),
            'significant_pairs': sum(1 for c in valid_human_correlations if c['significant']),
            'proportion_significant': sum(1 for c in valid_human_correlations if c['significant']) / len(valid_human_correlations) if valid_human_correlations else 0
        },
        'comparison_tests': {
            'mann_whitney': {
                'statistic': mw_statistic,
                'p_value': mw_pvalue,
                'significant': mw_pvalue < 0.05
            },
            'kolmogorov_smirnov': {
                'statistic': ks_statistic,
                'p_value': ks_pvalue,
                'significant': ks_pvalue < 0.05
            },
            't_test': {
                'statistic': t_statistic,
                'p_value': t_pvalue,
                'significant': t_pvalue < 0.05
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
            }
        }
    }

def main():
    print("=" * 80)
    print("CORRELATION P-VALUE ANALYSIS: LLMs vs HUMANS")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df_instruct, df_base = load_llm_data()
    df_human, question_cols, human_json = load_human_data()
    
    # Apply human data exclusion criteria
    print("Applying exclusion criteria to human data...")
    # Filter by completion time
    median_duration = df_human['Duration (in seconds)'].median()
    min_duration = 0.2 * median_duration
    df_human = df_human[df_human['Duration (in seconds)'] >= min_duration]
    
    # Filter attention check failures
    for group in range(1, 6):
        attention_col = f'Q{group}_8'
        if attention_col in df_human.columns:
            df_human = df_human[(df_human[attention_col].isna()) | (df_human[attention_col] == 100)]
    
    # Calculate LLM correlations with p-values
    print("\nCalculating LLM pairwise correlations with p-values...")
    llm_correlations = calculate_llm_correlations_with_pvalues(df_instruct, df_base)
    
    # Calculate human correlations with p-values
    print("Calculating human pairwise correlations with p-values...")
    human_correlations = calculate_human_correlations_with_pvalues(df_human, question_cols)
    
    # Compare distributions
    print("\nComparing correlation distributions...")
    comparison_results = compare_distributions(llm_correlations, human_correlations)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\nLLM CORRELATIONS:")
    print(f"  Number of model pairs: {comparison_results['llm_stats']['n_pairs']}")
    print(f"  Mean correlation: {comparison_results['llm_stats']['mean']:.4f}")
    print(f"  Std deviation: {comparison_results['llm_stats']['std']:.4f}")
    print(f"  Median correlation: {comparison_results['llm_stats']['median']:.4f}")
    print(f"  Significant correlations (p < 0.05): {comparison_results['llm_stats']['significant_pairs']} ({comparison_results['llm_stats']['proportion_significant']:.1%})")
    
    print("\nHUMAN CORRELATIONS:")
    print(f"  Number of rater pairs: {comparison_results['human_stats']['n_pairs']}")
    print(f"  Mean correlation: {comparison_results['human_stats']['mean']:.4f}")
    print(f"  Std deviation: {comparison_results['human_stats']['std']:.4f}")
    print(f"  Median correlation: {comparison_results['human_stats']['median']:.4f}")
    print(f"  Significant correlations (p < 0.05): {comparison_results['human_stats']['significant_pairs']} ({comparison_results['human_stats']['proportion_significant']:.1%})")
    
    print("\nSTATISTICAL COMPARISON (LLM vs Human distributions):")
    print(f"  Mann-Whitney U test:")
    print(f"    Statistic: {comparison_results['comparison_tests']['mann_whitney']['statistic']:.4f}")
    print(f"    P-value: {comparison_results['comparison_tests']['mann_whitney']['p_value']:.6f}")
    print(f"    Significant: {comparison_results['comparison_tests']['mann_whitney']['significant']}")
    
    print(f"  Kolmogorov-Smirnov test:")
    print(f"    Statistic: {comparison_results['comparison_tests']['kolmogorov_smirnov']['statistic']:.4f}")
    print(f"    P-value: {comparison_results['comparison_tests']['kolmogorov_smirnov']['p_value']:.6f}")
    print(f"    Significant: {comparison_results['comparison_tests']['kolmogorov_smirnov']['significant']}")
    
    print(f"  T-test:")
    print(f"    Statistic: {comparison_results['comparison_tests']['t_test']['statistic']:.4f}")
    print(f"    P-value: {comparison_results['comparison_tests']['t_test']['p_value']:.6f}")
    print(f"    Significant: {comparison_results['comparison_tests']['t_test']['significant']}")
    
    print(f"  Effect size (Cohen's d): {comparison_results['comparison_tests']['effect_size']['cohens_d']:.4f} ({comparison_results['comparison_tests']['effect_size']['interpretation']})")
    
    # Show some example significant and non-significant correlations
    print("\n" + "=" * 80)
    print("EXAMPLE LLM CORRELATIONS")
    print("=" * 80)
    
    # Sort by p-value
    llm_sorted = sorted(llm_correlations, key=lambda x: x['p_value'])
    
    print("\nMost significant LLM correlations:")
    for i, corr in enumerate(llm_sorted[:5]):
        model1 = corr['model1'].split('/')[-1] if '/' in corr['model1'] else corr['model1']
        model2 = corr['model2'].split('/')[-1] if '/' in corr['model2'] else corr['model2']
        print(f"  {model1[:20]} vs {model2[:20]}: r={corr['correlation']:.3f}, p={corr['p_value']:.6f}")
    
    print("\nLeast significant LLM correlations:")
    for i, corr in enumerate(llm_sorted[-5:]):
        model1 = corr['model1'].split('/')[-1] if '/' in corr['model1'] else corr['model1']
        model2 = corr['model2'].split('/')[-1] if '/' in corr['model2'] else corr['model2']
        print(f"  {model1[:20]} vs {model2[:20]}: r={corr['correlation']:.3f}, p={corr['p_value']:.6f}")
    
    # Save detailed results (convert numpy types to native Python types)
    def convert_to_native(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    results = {
        'llm_correlations': convert_to_native(llm_correlations),
        'human_correlations': convert_to_native(human_correlations),
        'comparison': convert_to_native(comparison_results)
    }
    
    with open('correlation_pvalues_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Results saved to correlation_pvalues_analysis.json")
    print("=" * 80)
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # LLM correlation distribution
        llm_corr_values = [c['correlation'] for c in llm_correlations]
        axes[0, 0].hist(llm_corr_values, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(llm_corr_values), color='red', linestyle='--', label=f'Mean: {np.mean(llm_corr_values):.3f}')
        axes[0, 0].set_xlabel('Correlation Coefficient')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('LLM Pairwise Correlations')
        axes[0, 0].legend()
        
        # Human correlation distribution
        human_corr_values = [c['correlation'] for c in human_correlations]
        axes[0, 1].hist(human_corr_values, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(np.mean(human_corr_values), color='red', linestyle='--', label=f'Mean: {np.mean(human_corr_values):.3f}')
        axes[0, 1].set_xlabel('Correlation Coefficient')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Human Pairwise Correlations')
        axes[0, 1].legend()
        
        # P-value distributions
        llm_pvalues = [c['p_value'] for c in llm_correlations]
        axes[1, 0].hist(llm_pvalues, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0.05, color='red', linestyle='--', label='p = 0.05')
        axes[1, 0].set_xlabel('P-value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('LLM Correlation P-values')
        axes[1, 0].legend()
        
        human_pvalues = [c['p_value'] for c in human_correlations]
        axes[1, 1].hist(human_pvalues, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].axvline(0.05, color='red', linestyle='--', label='p = 0.05')
        axes[1, 1].set_xlabel('P-value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Human Correlation P-values')
        axes[1, 1].legend()
        
        plt.suptitle('Correlation Analysis: LLMs vs Humans', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_pvalue_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualization saved to correlation_pvalue_distributions.png")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")

if __name__ == "__main__":
    main()