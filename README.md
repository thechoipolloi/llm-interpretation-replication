# Large Language Models as Unreliable Judges - Replication Code

This repository contains the replication code for the paper analyzing the reliability of Large Language Models (LLMs) as judges for legal interpretation tasks.

## Repository Structure

```
llm_interpretation_replication/
├── analysis/                 # Core analysis scripts
│   ├── analyze_perturbation_results.py     # Analyze prompt perturbation experiments
│   ├── analyze_results_base_versus_instruct.py  # Compare base vs instruction-tuned models
│   ├── calculate_cohens_kappa.py           # Calculate inter-rater agreement
│   ├── compare_base_vs_instruct.py         # Base vs instruct model comparisons
│   ├── compare_instruct_models.py          # Compare different instruction-tuned models
│   ├── config.py                            # Configuration settings
│   ├── model_comparison_graph.py           # Generate model comparison visualizations
│   └── perturb_prompts.py                  # Run prompt perturbation experiments
├── survey_analysis/          # Human survey analysis
│   ├── survey_analysis_consolidated.py     # Main survey analysis
│   ├── analyze_base_vs_instruct_vs_human.py # Compare models with human judgments
│   ├── analyze_llm_human_agreement.py      # LLM-human agreement analysis
│   ├── analyze_llm_human_agreement_bootstrap.py # Bootstrap confidence intervals
│   ├── analyze_llm_agreement_simple_bootstrap.py # Simple bootstrap analysis
│   ├── analyze_model_family_differences.py # Analyze differences between model families
│   ├── bootstrap_confidence_intervals.py   # Bootstrap CI calculations
│   └── calculate_correlation_pvalues.py    # Statistical significance tests
├── data/                     # Input data files
│   ├── word_meaning_survey_results.csv     # Human survey responses
│   ├── demographic_data.csv                # Survey participant demographics
│   ├── model_comparison_results.csv        # Model comparison outputs
│   └── instruct_model_comparison_results.csv # Instruction-tuned model results
├── results/                  # Output directory for generated results
└── requirements.txt          # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd llm_interpretation_replication
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (required for running model experiments):

   **Method 1: Using .env file (Recommended)**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your actual API keys
   nano .env  # or use your preferred editor
   ```

   **Method 2: Export as environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

   **Important:** Never commit your `.env` file with actual API keys to version control!

## Replication Instructions

### 1. Prompt Perturbation Analysis

To replicate the prompt sensitivity experiments:

```bash
# Run perturbation experiments (requires API access)
python analysis/perturb_prompts.py

# Analyze perturbation results
python analysis/analyze_perturbation_results.py
```

This will generate:
- Distribution plots of relative probabilities across prompts
- Statistical tests for normality
- Confidence interval visualizations

### 2. Model Comparison Analysis

To compare base vs instruction-tuned models:

```bash
# Run model comparisons
python analysis/compare_base_vs_instruct.py

# Analyze results
python analysis/analyze_results_base_versus_instruct.py

# Generate comparison graphs
python analysis/model_comparison_graph.py
```

Output includes:
- Correlation matrices between models
- Distribution of probability differences
- Visualization of model agreement/disagreement

### 3. Inter-Model Agreement (Cohen's Kappa)

```bash
python analysis/calculate_cohens_kappa.py
```

This calculates:
- Pairwise Cohen's kappa coefficients
- Overall inter-rater reliability metrics
- Agreement matrices

### 4. Human Survey Analysis

To replicate the human survey comparison:

```bash
# Main survey analysis
python survey_analysis/survey_analysis_consolidated.py

# Compare with LLM results
python survey_analysis/analyze_base_vs_instruct_vs_human.py

# Calculate LLM-human agreement
python survey_analysis/analyze_llm_human_agreement.py

# Bootstrap confidence intervals
python survey_analysis/analyze_llm_human_agreement_bootstrap.py

# Model family differences
python survey_analysis/analyze_model_family_differences.py
```

Results include:
- Human-LLM correlation coefficients
- Bootstrap confidence intervals
- Statistical significance tests
- Model family clustering analysis

### 5. Statistical Significance Testing

```bash
# Calculate p-values for correlations
python survey_analysis/calculate_correlation_pvalues.py

# Bootstrap confidence intervals
python survey_analysis/bootstrap_confidence_intervals.py
```

## Data Files

### Input Data

- `word_meaning_survey_results.csv`: Human survey responses for word meaning judgments
- `demographic_data.csv`: Demographic information of survey participants
- `model_comparison_results.csv`: Pre-computed model outputs (if not regenerating)
- `instruct_model_comparison_results.csv`: Instruction-tuned model outputs

### Output Files

All generated results will be saved in the `results/` directory, including:
- Visualization plots (PNG/PDF)
- Statistical analysis results (JSON/CSV)
- Correlation matrices
- Bootstrap confidence intervals

## Key Findings Replicated

1. **Prompt Sensitivity**: LLMs show high sensitivity to minor prompt variations
2. **Inter-Model Disagreement**: Low correlation between different model families
3. **Instruction Tuning Effects**: Significant differences between base and instruction-tuned models
4. **Human-LLM Divergence**: Weak correlation between human judgments and LLM outputs

## Configuration

Edit `analysis/config.py` to modify:
- Model selection
- API endpoints
- Number of bootstrap iterations
- Statistical significance thresholds
- Output directory paths

## Troubleshooting

1. **API Rate Limits**: If encountering rate limits, add delays in `perturb_prompts.py`
2. **Memory Issues**: For large-scale bootstrap, reduce `n_bootstrap` in configuration
3. **Missing Data**: Ensure all CSV files are in the `data/` directory

## Citation

If you use this code, please cite:

```bibtex
@article{choi2025llm,
  title={Large Language Models as Unreliable Judges},
  author={Choi, Jonathan H.},
  journal={[Journal Name]},
  year={2025}
}
```

## License

This code is released under the MIT License. See LICENSE file for details.

## Contact

For questions or issues with the replication code, please open an issue on GitHub or contact [contact information].