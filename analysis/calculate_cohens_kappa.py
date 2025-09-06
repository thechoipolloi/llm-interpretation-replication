import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy import stats
import os

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

def load_model_comparison_data():
    """
    Load the results from instruct_model_comparison_results.csv
    Returns a DataFrame with model comparison data
    """
    try:
        # Try to load from standard directory
        base_dir = "G:/My Drive/Computational/llm_interpretation"
        file_path = Path(base_dir) / "instruct_model_comparison_results.csv"
        
        if not file_path.exists():
            # If not found, try current directory
            file_path = Path("instruct_model_comparison_results.csv")
        
        if file_path.exists():
            print(f"Loading model comparison data from {file_path}")
            df = pd.read_csv(file_path)
            return df
        else:
            print("Model comparison data file not found.")
            return None
    except Exception as e:
        print(f"Error loading model comparison data: {e}")
        return None

def load_perturbation_data():
    """
    Load the results from combined_results.xlsx or results.xlsx
    Returns a DataFrame with prompt perturbation data
    """
    try:
        # Try to load from standard directory
        base_dir = "G:/My Drive/Computational/llm_interpretation"
        file_path = Path(base_dir) / "combined_results.xlsx"
        
        if not file_path.exists():
            # Try results.xlsx instead
            file_path = Path(base_dir) / "results.xlsx"
        
        if not file_path.exists():
            # If not found, try current directory
            file_path = Path("combined_results.xlsx")
            if not file_path.exists():
                file_path = Path("results.xlsx")
        
        if file_path.exists():
            print(f"Loading prompt perturbation data from {file_path}")
            df = pd.read_excel(file_path)
            return df
        else:
            print("Perturbation data file not found.")
            return None
    except Exception as e:
        print(f"Error loading perturbation data: {e}")
        return None

def prepare_model_data(df):
    """
    Prepare the model comparison data for analysis
    Convert relative probabilities to binary decisions and calculate Cohen's kappa
    """
    if df is None:
        return None
    
    print("Preparing model comparison data...")
    
    # Calculate binary decisions based on relative probability
    # If relative_prob > 0.5, the model is choosing "Yes", otherwise "No"
    df['binary_decision'] = df['relative_prob'].apply(lambda x: 1 if x > 0.5 else 0)
    
    # Group data by prompt
    prompt_groups = df.groupby('prompt')
    
    # Store kappa metrics for each prompt
    kappa_results = []
    
    # Calculate kappa for each prompt across models
    for prompt, group in prompt_groups:
        # Skip if fewer than 2 models
        if len(group) < 2:
            continue
        
        # Calculate kappa for all model pairs
        models = group['model'].unique()
        n_models = len(models)
        
        if n_models < 2:
            continue
        
        # Create a model-indexed Series of binary decisions
        decisions = group.set_index('model')['binary_decision']
        
        # Calculate all pairwise kappas
        kappa_pairs = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                model_i = models[i]
                model_j = models[j]
                try:
                    # Get decisions for both models
                    decision_i = decisions[model_i]
                    decision_j = decisions[model_j]
                    
                    # Calculate Cohen's kappa
                    kappa = cohen_kappa_score(
                        [decision_i], 
                        [decision_j]
                    )
                    kappa_pairs.append(kappa)
                except Exception as e:
                    print(f"Error calculating kappa for models {model_i} and {model_j}: {e}")
        
        # Average all pairwise kappas for this prompt
        if kappa_pairs:
            avg_kappa = np.mean(kappa_pairs)
            kappa_results.append({
                'prompt': prompt,
                'avg_pairwise_kappa': avg_kappa,
                'n_models': n_models,
                'min_kappa': np.min(kappa_pairs),
                'max_kappa': np.max(kappa_pairs),
                'std_kappa': np.std(kappa_pairs),
                'agree_percent': group['binary_decision'].mean() if group['binary_decision'].mean() > 0.5 else 1 - group['binary_decision'].mean()
            })
    
    return pd.DataFrame(kappa_results)

def prepare_perturbation_data(df):
    """
    Prepare the perturbation data for analysis
    Convert relative probabilities to binary decisions and calculate metrics
    """
    if df is None:
        return None
    
    print("Preparing prompt perturbation data...")
    
    # Ensure we have total probabilities and relative probabilities
    if 'Total_Prob' not in df.columns:
        df['Total_Prob'] = df['Token_1_Prob'] + df['Token_2_Prob']
    
    if 'Relative_Prob' not in df.columns:
        df['Relative_Prob'] = df['Token_1_Prob'] / df['Total_Prob']
    
    # Calculate binary decisions based on relative probability
    # If Relative_Prob > 0.5, the model is choosing the first token, otherwise the second token
    df['binary_decision'] = df['Relative_Prob'].apply(lambda x: 1 if x > 0.5 else 0)
    
    # Group by original prompt
    prompt_groups = df.groupby('Original Main Part')
    
    # Store results for each prompt
    perturbation_results = []
    
    for prompt, group in prompt_groups:
        # Calculate metrics
        n_variations = len(group)
        agree_percent = group['binary_decision'].mean() if group['binary_decision'].mean() > 0.5 else 1 - group['binary_decision'].mean()
        
        # Calculate a "self-kappa" for this prompt's variations
        # This measures how consistent the model is with itself across perturbations
        decisions = group['binary_decision'].values
        
        # Create synthetic paired observations from perturbations
        # We'll use bootstrap resampling to create pairs
        np.random.seed(42)  # For reproducibility
        n_bootstraps = 1000
        bootstrap_kappas = []
        
        for _ in range(n_bootstraps):
            # Create two bootstrap samples from the decisions
            idx1 = np.random.choice(len(decisions), size=len(decisions), replace=True)
            idx2 = np.random.choice(len(decisions), size=len(decisions), replace=True)
            
            sample1 = decisions[idx1]
            sample2 = decisions[idx2]
            
            # Calculate Cohen's kappa between these samples
            try:
                kappa = cohen_kappa_score(sample1, sample2)
                bootstrap_kappas.append(kappa)
            except Exception as e:
                pass
        
        # Calculate mean kappa from bootstrap samples
        if bootstrap_kappas:
            mean_kappa = np.mean(bootstrap_kappas)
            std_kappa = np.std(bootstrap_kappas)
            perturbation_results.append({
                'prompt': prompt,
                'n_variations': n_variations,
                'agree_percent': agree_percent,
                'self_kappa': mean_kappa,
                'self_kappa_std': std_kappa,
                'min_kappa': np.min(bootstrap_kappas),
                'max_kappa': np.max(bootstrap_kappas)
            })
    
    return pd.DataFrame(perturbation_results)

def get_interpretation_prompt_data(model_df, pert_df):
    """
    Match legal interpretation prompts between model comparison and perturbation data
    """
    if model_df is None or pert_df is None:
        return None, None
    
    print("\nMatching legal interpretation prompts between datasets...")
    
    # Common legal interpretation prompts - these need to be mapped between datasets
    legal_prompts = {
        "Insurance Policy Water Damage Exclusion": 
            ["water damage", "levee", "flood", "insurance policy"],
        "Prenuptial Agreement Petition Filing Date": 
            ["prenuptial", "petition", "dissolution", "marriage", "filing"],
        "Contract Term Affiliate Interpretation": 
            ["contract", "affiliate", "royalty", "1961", "company"],
        "Construction Payment Terms Interpretation": 
            ["contractor", "usual manner", "payment", "foundry", "construction"],
        "Insurance Policy Burglary Coverage": 
            ["insurance", "felonious", "burglary", "theft", "visible marks"]
    }
    
    # Debug what prompts are available in each dataset
    print(f"Model dataset has {len(model_df)} prompts")
    print(f"Perturbation dataset has {len(pert_df)} prompts")
    
    # First find matches in model comparison dataset
    print("\nSearching for matches in model comparison dataset...")
    model_legal_data = []
    for title, keywords in legal_prompts.items():
        found_match = False
        for keyword in keywords:
            if not found_match:
                # Try pattern matching with keyword
                matches = model_df[model_df['prompt'].str.contains(keyword, case=False, regex=False, na=False)]
                if not matches.empty:
                    print(f"Found {len(matches)} matches for '{title}' using keyword '{keyword}'")
                    for _, row in matches.iterrows():
                        # Only add if we haven't already found this prompt
                        if not any(d.get('prompt') == row['prompt'] for d in model_legal_data):
                            print(f"  Adding prompt: {row['prompt'][:50]}...")
                            model_legal_data.append({
                                'title': title,
                                'prompt': row['prompt'],
                                'avg_pairwise_kappa': row['avg_pairwise_kappa'],
                                'n_models': row['n_models'],
                                'agree_percent': row['agree_percent'],
                                'source': 'model_comparison'
                            })
                            found_match = True
                            break
    
    # Then find matches in perturbation dataset
    print("\nSearching for matches in perturbation dataset...")
    pert_legal_data = []
    
    # Check what columns are available in perturbation dataset
    if len(pert_df) > 0:
        print(f"Perturbation dataset columns: {pert_df.columns.tolist()}")
    
    # Try to match using different prompt columns
    prompt_columns = ['prompt', 'Original Main Part', 'Full Rephrased Prompt']
    
    for title, keywords in legal_prompts.items():
        found_match = False
        for col in prompt_columns:
            if not found_match and col in pert_df.columns:
                for keyword in keywords:
                    # Try pattern matching with keyword
                    matches = pert_df[pert_df[col].str.contains(keyword, case=False, regex=False, na=False)]
                    if not matches.empty:
                        print(f"Found {len(matches)} matches for '{title}' using keyword '{keyword}' in column '{col}'")
                        for _, row in matches.iterrows():
                            # Use the matched column as the prompt
                            prompt_text = row[col]
                            # Only add if we haven't already found this prompt title
                            if not any(d.get('title') == title for d in pert_legal_data):
                                print(f"  Adding prompt: {prompt_text[:50]}...")
                                pert_legal_data.append({
                                    'title': title,
                                    'prompt': prompt_text,
                                    'self_kappa': row['self_kappa'],
                                    'n_variations': row['n_variations'],
                                    'agree_percent': row['agree_percent'],
                                    'source': 'perturbation'
                                })
                                found_match = True
                                break
                    if found_match:
                        break
    
    # Check if we found matches
    if not model_legal_data:
        print("Warning: No matches found in model comparison dataset")
    
    if not pert_legal_data:
        print("Warning: No matches found in perturbation dataset")
    
    # Convert to DataFrames
    model_legal_df = pd.DataFrame(model_legal_data) if model_legal_data else pd.DataFrame()
    pert_legal_df = pd.DataFrame(pert_legal_data) if pert_legal_data else pd.DataFrame()
    
    # Log the matches we found
    print(f"\nFound {len(model_legal_df)} matches in model dataset and {len(pert_legal_df)} matches in perturbation dataset")
    
    return model_legal_df, pert_legal_df

def calculate_combined_kappa(model_kappa, perturbation_kappa, model_kappa_std=0.1, pert_kappa_std=0.1, bootstrap=True):
    """
    Calculate combined kappa statistics considering both sources of variation
    Uses bootstrap to estimate the distribution of possible kappas
    
    Parameters:
    model_kappa: The mean kappa value from model variation
    perturbation_kappa: The mean kappa value from prompt perturbation
    model_kappa_std: Standard deviation of model kappa (default 0.1)
    pert_kappa_std: Standard deviation of perturbation kappa (default 0.1)
    bootstrap: Whether to use bootstrap for CI estimation (default True)
    """
    if bootstrap:
        # Use bootstrap to estimate the combined kappa distribution
        n_bootstraps = 1000
        combined_kappas = []
        
        # Assuming independence between model variation and perturbation variation
        # We randomly sample from both distributions and combine
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_bootstraps):
            # Sample a model kappa and a perturbation kappa
            model_sample = model_kappa + np.random.normal(0, model_kappa_std)
            pert_sample = perturbation_kappa + np.random.normal(0, pert_kappa_std)
            
            # Combine the two sources of variation (approximate)
            # Lower kappa means more disagreement, we use the minimum
            combined_kappa = min(model_sample, pert_sample)
            combined_kappas.append(combined_kappa)
        
        # Calculate statistics from bootstrap samples
        mean_kappa = np.mean(combined_kappas)
        median_kappa = np.median(combined_kappas)
        lower_ci = np.percentile(combined_kappas, 2.5)
        upper_ci = np.percentile(combined_kappas, 97.5)
        
        return {
            'mean_kappa': mean_kappa,
            'median_kappa': median_kappa,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'bootstrap_samples': combined_kappas
        }
    else:
        # Simple minimum approach - take the minimum kappa as the combined kappa
        combined_kappa = min(model_kappa, perturbation_kappa)
        return {
            'combined_kappa': combined_kappa
        }

def interpret_kappa(kappa):
    """
    Provide an interpretation of the kappa value based on common benchmarks
    """
    if kappa < 0:
        return "Poor agreement (worse than chance)"
    elif kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Moderate agreement"
    elif kappa < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def create_plots(model_data, pert_data, combined_results, output_dir):
    """
    Create visualizations for the kappa statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Bar chart comparing model kappa vs. perturbation kappa
    plt.figure(figsize=(14, 10))
    
    # Prepare data for plotting
    titles = []
    model_kappas = []
    pert_kappas = []
    combined_kappas = []
    
    for title, results in combined_results.items():
        titles.append(title)
        model_kappas.append(results['model_kappa'])
        pert_kappas.append(results['perturbation_kappa'])
        combined_kappas.append(results['combined']['mean_kappa'])
    
    x = np.arange(len(titles))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width, model_kappas, width, label='Model Variation Kappa')
    rects2 = ax.bar(x, pert_kappas, width, label='Perturbation Kappa')
    rects3 = ax.bar(x + width, combined_kappas, width, label='Combined Kappa')
    
    # Add labels and title
    ax.set_ylabel('Cohen\'s Kappa Value')
    ax.set_title('Comparison of Kappa Values by Source of Variation')
    ax.set_xticks(x)
    ax.set_xticklabels([t.split()[-2:] for t in titles], rotation=45, ha='right')
    ax.legend()
    
    # Add horizontal lines for interpretation ranges
    kappa_ranges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    kappa_labels = ["Poor", "Slight", "Fair", "Moderate", "Substantial", "Perfect"]
    
    for i in range(len(kappa_ranges) - 1):
        plt.axhline(y=kappa_ranges[i], color='gray', linestyle='--', alpha=0.5)
        ax.text(-0.5, (kappa_ranges[i] + kappa_ranges[i+1])/2, kappa_labels[i], 
                verticalalignment='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kappa_comparison_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Combined kappa distribution for each prompt (from bootstrap samples)
    for title, results in combined_results.items():
        plt.figure(figsize=(10, 6))
        bootstrap_samples = results['combined']['bootstrap_samples']
        
        # Kernel density plot of the combined kappa distribution
        sns.histplot(bootstrap_samples, kde=True)
        
        # Add vertical lines for mean and 95% CI
        plt.axvline(x=results['combined']['mean_kappa'], color='r', linestyle='--', 
                   label=f'Mean: {results["combined"]["mean_kappa"]:.3f}')
        plt.axvline(x=results['combined']['lower_ci'], color='g', linestyle=':', 
                   label=f'2.5th percentile: {results["combined"]["lower_ci"]:.3f}')
        plt.axvline(x=results['combined']['upper_ci'], color='g', linestyle=':', 
                   label=f'97.5th percentile: {results["combined"]["upper_ci"]:.3f}')
        
        # Add interpretation annotation
        mean_interp = interpret_kappa(results['combined']['mean_kappa'])
        plt.text(0.5, 0.9, f"Interpretation: {mean_interp}", 
                 horizontalalignment='center', transform=plt.gca().transAxes, fontsize=14,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Add horizontal lines for interpretation ranges
        for i in range(len(kappa_ranges) - 1):
            plt.axvline(x=kappa_ranges[i], color='gray', linestyle='--', alpha=0.2)
        
        plt.xlabel('Cohen\'s Kappa Value')
        plt.ylabel('Frequency')
        plt.title(f'Bootstrap Distribution of Combined Kappa: {title}')
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        short_title = '_'.join(title.split()[-2:]).lower()
        plt.savefig(f"{output_dir}/kappa_distribution_{short_title}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Scatterplot of model kappa vs. perturbation kappa
    plt.figure(figsize=(10, 8))
    plt.scatter(model_kappas, pert_kappas, s=100, alpha=0.7)
    
    # Add diagonal line for reference
    min_val = min(min(model_kappas), min(pert_kappas))
    max_val = max(max(model_kappas), max(pert_kappas))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add labels for each point
    for i, title in enumerate(titles):
        short_label = ' '.join(title.split()[-2:])
        plt.annotate(short_label, (model_kappas[i], pert_kappas[i]), 
                    fontsize=12, xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Model Variation Kappa')
    plt.ylabel('Perturbation Kappa')
    plt.title('Model Variation vs. Prompt Perturbation Kappa')
    
    # Create equal aspect ratio
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal and vertical lines for interpretation ranges
    for val in [0.2, 0.4, 0.6, 0.8]:
        plt.axhline(y=val, color='gray', linestyle='--', alpha=0.2)
        plt.axvline(x=val, color='gray', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kappa_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the analysis
    """
    print("Loading data...")
    # Load data from both sources
    model_comparison_df = load_model_comparison_data()
    perturbation_df = load_perturbation_data()
    
    if model_comparison_df is None or perturbation_df is None:
        print("Could not load one or both data sources. Exiting.")
        return
    
    # Prepare data for analysis
    model_kappa_df = prepare_model_data(model_comparison_df)
    perturbation_kappa_df = prepare_perturbation_data(perturbation_df)
    
    if model_kappa_df is None or perturbation_kappa_df is None:
        print("Could not prepare kappa metrics. Exiting.")
        return
    
    # Get legal interpretation prompts data
    model_legal_df, pert_legal_df = get_interpretation_prompt_data(model_kappa_df, perturbation_kappa_df)
    
    # Check if legal data was properly matched
    if model_legal_df is None or pert_legal_df is None or model_legal_df.empty or pert_legal_df.empty:
        print("Could not match legal interpretation prompts between datasets. Exiting.")
        return
    
    # Verify that the legal dataframes have the expected columns
    required_model_cols = ['title', 'prompt', 'avg_pairwise_kappa', 'n_models', 'agree_percent']
    required_pert_cols = ['title', 'prompt', 'self_kappa', 'n_variations', 'agree_percent']
    
    if not all(col in model_legal_df.columns for col in required_model_cols):
        print(f"Model legal dataframe is missing required columns. Found: {model_legal_df.columns.tolist()}")
        return
    
    if not all(col in pert_legal_df.columns for col in required_pert_cols):
        print(f"Perturbation legal dataframe is missing required columns. Found: {pert_legal_df.columns.tolist()}")
        return
    
    # Create output directory
    output_dir = Path("output/kappa_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data
    model_kappa_df.to_csv(output_dir / "model_kappa_metrics.csv", index=False)
    perturbation_kappa_df.to_csv(output_dir / "perturbation_kappa_metrics.csv", index=False)
    model_legal_df.to_csv(output_dir / "model_legal_kappas.csv", index=False)
    pert_legal_df.to_csv(output_dir / "perturbation_legal_kappas.csv", index=False)
    
    # Calculate combined kappa for each legal interpretation prompt
    combined_results = {}
    
    # For each legal prompt title
    for title in model_legal_df['title'].unique():
        model_data = model_legal_df[model_legal_df['title'] == title]
        pert_data = pert_legal_df[pert_legal_df['title'] == title]
        
        if not model_data.empty and not pert_data.empty:
            # Get kappa statistics
            model_kappa = model_data['avg_pairwise_kappa'].mean()
            model_kappa_std = model_data['avg_pairwise_kappa'].std() if len(model_data) > 1 else 0.1
            
            pert_kappa = pert_data['self_kappa'].mean()
            pert_kappa_std = pert_data['self_kappa'].std() if len(pert_data) > 1 else 0.1
            
            # Calculate combined kappa
            combined = calculate_combined_kappa(model_kappa, pert_kappa, model_kappa_std, pert_kappa_std)
            
            # Interpret the kappa values
            model_interp = interpret_kappa(model_kappa)
            pert_interp = interpret_kappa(pert_kappa)
            combined_interp = interpret_kappa(combined['mean_kappa'])
            
            # Store results
            combined_results[title] = {
                'model_kappa': model_kappa,
                'model_kappa_std': model_kappa_std,
                'model_interpretation': model_interp,
                'perturbation_kappa': pert_kappa,
                'perturbation_kappa_std': pert_kappa_std,
                'perturbation_interpretation': pert_interp,
                'combined': combined,
                'combined_interpretation': combined_interp
            }
    
    # Check if we were able to calculate any combined results
    if not combined_results:
        print("No combined kappa results were calculated. This could be due to no matching prompts between datasets.")
        return
    
    # Save combined results
    combined_df = []
    for title, results in combined_results.items():
        combined_df.append({
            'Prompt': title,
            'Model Kappa': results['model_kappa'],
            'Model Kappa Std': results['model_kappa_std'],
            'Model Interpretation': results['model_interpretation'],
            'Perturbation Kappa': results['perturbation_kappa'],
            'Perturbation Kappa Std': results['perturbation_kappa_std'],
            'Perturbation Interpretation': results['perturbation_interpretation'],
            'Combined Mean Kappa': results['combined']['mean_kappa'],
            'Combined Median Kappa': results['combined']['median_kappa'],
            'Combined Lower CI': results['combined']['lower_ci'],
            'Combined Upper CI': results['combined']['upper_ci'],
            'Combined Interpretation': results['combined_interpretation']
        })
    
    # Save to CSV
    combined_results_df = pd.DataFrame(combined_df)
    combined_results_df.to_csv(output_dir / "combined_kappa_results.csv", index=False)
    
    # Create LaTeX table
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Cohen's Kappa Analysis of Model Variation vs. Prompt Perturbation}\n"
    latex_table += "\\label{tab:kappa_analysis}\n"
    latex_table += "\\begin{tabular}{lccccc}\n"
    latex_table += "\\hline\n"
    latex_table += "Prompt & Model $\\kappa$ & Perturbation $\\kappa$ & Combined $\\kappa$ & 95\\% CI & Interpretation \\\\ \n"
    latex_table += "\\hline\n"
    
    for _, row in combined_results_df.iterrows():
        # Extract prompt name - just the last two words
        prompt_parts = row['Prompt'].split()
        short_prompt = " ".join(prompt_parts[-2:])
        
        # Format values with proper precision
        model_kappa = f"{row['Model Kappa']:.3f}"
        pert_kappa = f"{row['Perturbation Kappa']:.3f}"
        combined_kappa = f"{row['Combined Mean Kappa']:.3f}"
        ci = f"[{row['Combined Lower CI']:.3f}, {row['Combined Upper CI']:.3f}]"
        interp = row['Combined Interpretation']
        
        latex_table += f"{short_prompt} & {model_kappa} & {pert_kappa} & {combined_kappa} & {ci} & {interp} \\\\ \n"
    
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    with open(output_dir / "kappa_analysis_table.tex", "w") as f:
        f.write(latex_table)
    
    # Create plots
    create_plots(model_legal_df, pert_legal_df, combined_results, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}")
    
    # Print summary of findings
    print("\nSummary of Cohen's Kappa Analysis:")
    print("---------------------------------")
    for title, results in combined_results.items():
        print(f"\n{title}:")
        print(f"  Model Variation Kappa: {results['model_kappa']:.3f} ({results['model_interpretation']})")
        print(f"  Prompt Perturbation Kappa: {results['perturbation_kappa']:.3f} ({results['perturbation_interpretation']})")
        print(f"  Combined Kappa: {results['combined']['mean_kappa']:.3f} ({results['combined_interpretation']})")
        print(f"  95% CI: [{results['combined']['lower_ci']:.3f}, {results['combined']['upper_ci']:.3f}]")

if __name__ == "__main__":
    main() 