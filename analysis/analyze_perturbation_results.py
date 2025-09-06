import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats  # Renamed to avoid conflicts
import json
import ast

# Set global font sizes for all plots - INCREASED SIZES
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Function to conduct normality tests on distribution data
def conduct_normality_tests(data, column_name, prompt_idx):
    """
    Conduct Kolmogorov-Smirnov and Anderson-Darling tests for normality.
    Returns a dictionary with test results.
    """
    # Extract the data
    values = data[column_name].values
    
    # Filter out non-finite values (NaN, inf, -inf)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    
    # Check if we have enough valid data
    if len(values) == 0:
        print(f"Warning: No finite values found for prompt {prompt_idx + 1}, column {column_name}")
        return {
            'Prompt': prompt_idx + 1,
            'Distribution Mean': np.nan,
            'Distribution Std Dev': np.nan,
            'KS Statistic': np.nan,
            'KS p-value': np.nan,
            'KS Normal (p>0.05)': False,
            'AD Statistic': np.nan,
            'AD p-value': np.nan,
            'AD Critical Value (5%)': np.nan,
            'AD Normal (stat<crit)': False
        }
    
    if len(values) < 3:
        print(f"Warning: Insufficient data for normality tests (n={len(values)}) for prompt {prompt_idx + 1}, column {column_name}")
        return {
            'Prompt': prompt_idx + 1,
            'Distribution Mean': np.mean(values) if len(values) > 0 else np.nan,
            'Distribution Std Dev': np.std(values) if len(values) > 1 else np.nan,
            'KS Statistic': np.nan,
            'KS p-value': np.nan,
            'KS Normal (p>0.05)': False,
            'AD Statistic': np.nan,
            'AD p-value': np.nan,
            'AD Critical Value (5%)': np.nan,
            'AD Normal (stat<crit)': False
        }
    
    # Fit a normal distribution to the data
    mu, sigma = scipy_stats.norm.fit(values)
    
    # Conduct Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = scipy_stats.kstest(values, 'norm', args=(mu, sigma))
    
    # Conduct Anderson-Darling test
    ad_result = scipy_stats.anderson(values, 'norm')
    ad_statistic = ad_result.statistic
    ad_critical_values = ad_result.critical_values
    ad_significance_level = ad_result.significance_level
    
    # Determine if AD test indicates normality (at 5% significance)
    ad_normal = ad_statistic < ad_critical_values[2]  # Index 2 corresponds to 5% significance
    
    # Calculate p-value for Anderson-Darling test
    # We'll use an approximation as scipy doesn't directly provide AD p-values
    # The approximation is based on the ratio of the statistic to the critical value at 5%
    ad_pvalue = np.exp(-0.5 * ad_statistic)  # Simple approximation
    if ad_statistic > 10:
        ad_pvalue = 0.0001  # Very small p-value for large statistics
    elif ad_statistic > ad_critical_values[4]:  # Index 4 corresponds to 1% significance
        ad_pvalue = 0.005  # Between 0.5% and 1%
    elif ad_statistic > ad_critical_values[3]:  # Index 3 corresponds to 2.5% significance
        ad_pvalue = 0.015  # Between 1% and 2.5%
    elif ad_statistic > ad_critical_values[2]:  # Index 2 corresponds to 5% significance
        ad_pvalue = 0.035  # Between 2.5% and 5%
    elif ad_statistic > ad_critical_values[1]:  # Index 1 corresponds to 10% significance
        ad_pvalue = 0.075  # Between 5% and 10%
    else:
        ad_pvalue = 0.15  # Greater than 10% significance
    
    # Create a results dictionary
    results = {
        'Prompt': prompt_idx + 1,
        'Distribution Mean': mu,
        'Distribution Std Dev': sigma,
        'KS Statistic': ks_statistic,
        'KS p-value': ks_pvalue,
        'KS Normal (p>0.05)': ks_pvalue > 0.05,
        'AD Statistic': ad_statistic,
        'AD p-value': ad_pvalue,
        'AD Critical Value (5%)': ad_critical_values[2],
        'AD Normal (stat<crit)': ad_normal
    }
    
    return results

# Function to conduct truncated normal distribution test accounting for zero-inflation and one-inflation
def conduct_truncated_normal_test(data, column_name, prompt_idx, n_simulations=100000):
    """
    Test fit of a truncated normal distribution that accounts for zero-inflation and one-inflation
    by simulating what would happen if values outside [0,1] were constrained to 0 or 1.
    Uses Monte Carlo simulation to create a theoretical distribution with same mean and std dev.
    Returns a dictionary with test results.
    """
    # Extract the data
    values = data[column_name].values
    
    # Filter out non-finite values (NaN, inf, -inf)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    
    # Check if we have enough valid data
    if len(values) == 0:
        print(f"Warning: No finite values found for truncated normal test, prompt {prompt_idx + 1}, column {column_name}")
        return {
            'Prompt': prompt_idx + 1,
            'Column': column_name,
            'Model Type': 'Truncated Normal with Zero/One Inflation',
            'Model Fit': 'Failed - No finite values',
            'Zero Proportion': np.nan,
            'One Proportion': np.nan,
            'Interior Mean': np.nan,
            'Interior Std Dev': np.nan,
            'KS Statistic': np.nan,
            'KS p-value': np.nan,
            'AD Statistic': np.nan,
            'AD p-value': np.nan,
            'Model Adequate (KS p>0.05)': False,
            'Model Adequate (AD p>0.05)': False,
            'Model Adequate (Combined)': False
        }, np.array([])
    
    # Count zeros and ones (with a small tolerance)
    epsilon = 1e-6  # Small tolerance for floating point comparison
    n_zeros = np.sum(values < epsilon)
    n_ones = np.sum(values > 1 - epsilon)
    n_total = len(values)
    
    # Calculate proportion of zeros and ones
    zero_proportion = n_zeros / n_total
    one_proportion = n_ones / n_total
    
    # Calculate mean and std dev of non-zero, non-one values
    interior_values = values[(values >= epsilon) & (values <= 1 - epsilon)]
    if len(interior_values) == 0:
        # If all values are 0 or 1, we can't fit a truncated normal
        return {
            'Prompt': prompt_idx + 1,
            'Column': column_name,
            'Model Type': 'Truncated Normal with Zero/One Inflation',
            'Model Fit': 'Failed - All values are 0 or 1',
            'Zero Proportion': zero_proportion,
            'One Proportion': one_proportion,
            'Interior Mean': np.nan,
            'Interior Std Dev': np.nan,
            'KS Statistic': np.nan,
            'KS p-value': np.nan,
            'AD Statistic': np.nan,
            'AD p-value': np.nan,
            'Model Adequate (KS p>0.05)': False,
            'Model Adequate (AD p>0.05)': False,
            'Model Adequate (Combined)': False
        }
    
    interior_mean = np.mean(interior_values)
    interior_std = np.std(interior_values)
    
    # Generate a normal distribution with the same mean and std dev
    # We'll need to iterate to find parameters that result in the right mean and std dev after truncation
    target_mean = np.mean(values)
    target_std = np.std(values)
    
    # Start with the empirical mean and std dev as initial guesses
    mu_guess = target_mean
    sigma_guess = target_std
    
    # Function to simulate truncated normal with zero/one inflation
    def simulate_truncated_normal(mu, sigma, n_samples):
        # Generate normal samples
        normal_samples = np.random.normal(mu, sigma, n_samples)
        # Truncate to [0, 1]
        truncated_samples = np.clip(normal_samples, 0, 1)
        return truncated_samples
    
    # Optimize to find best mu and sigma for the underlying normal
    # This is a simple iterative approach - could be more sophisticated
    max_iterations = 30  # Increased from 20
    convergence_threshold = 0.0001  # Tighter convergence requirement
    
    for i in range(max_iterations):
        # Simulate with current parameters
        simulated = simulate_truncated_normal(mu_guess, sigma_guess, n_simulations)
        sim_mean = np.mean(simulated)
        sim_std = np.std(simulated)
        
        # Print progress for debugging
        if i % 5 == 0 or i == max_iterations - 1:
            print(f"Prompt {prompt_idx+1}, Iteration {i+1}: " +
                  f"Target mean={target_mean:.4f}, sim mean={sim_mean:.4f}, " +
                  f"Target std={target_std:.4f}, sim std={sim_std:.4f}")
        
        # Check if we've converged
        mean_diff = abs(sim_mean - target_mean)
        std_diff = abs(sim_std - target_std)
        
        if mean_diff < convergence_threshold and std_diff < convergence_threshold:
            print(f"Converged at iteration {i+1} for prompt {prompt_idx+1}")
            break
        
        # Adjust parameters based on difference
        # Use a damping factor to avoid overshooting
        damping = 0.5
        mean_adjustment = (target_mean / sim_mean) if sim_mean > 0 else 1
        std_adjustment = (target_std / sim_std) if sim_std > 0 else 1
        
        # Apply damping to adjustments
        mean_adjustment = 1 + damping * (mean_adjustment - 1)
        std_adjustment = 1 + damping * (std_adjustment - 1)
        
        # Update guesses
        mu_guess = mu_guess * mean_adjustment
        sigma_guess = sigma_guess * std_adjustment
        
        # Additional adjustment focused on improving the mean match
        # Shift the mean directly if needed
        if mean_diff > 0.001:
            mu_guess += damping * (target_mean - sim_mean)
    
    # Final simulation with optimized parameters
    optimized_simulation = simulate_truncated_normal(mu_guess, sigma_guess, n_simulations)
    final_sim_mean = np.mean(optimized_simulation)
    final_sim_std = np.std(optimized_simulation)
    
    # Verify if we achieved our target mean and std dev
    mean_accuracy = abs(final_sim_mean - target_mean) / target_mean if target_mean != 0 else abs(final_sim_mean)
    std_accuracy = abs(final_sim_std - target_std) / target_std if target_std != 0 else abs(final_sim_std)
    
    # Report accuracy
    print(f"Final verification for prompt {prompt_idx+1}:")
    print(f"  Target mean: {target_mean:.6f}, Simulated mean: {final_sim_mean:.6f}, Relative error: {mean_accuracy:.6f}")
    print(f"  Target std: {target_std:.6f}, Simulated std: {final_sim_std:.6f}, Relative error: {std_accuracy:.6f}")
    
    # If accuracy is poor, try a different approach
    if mean_accuracy > 0.01 or std_accuracy > 0.01:
        print(f"Warning: Poor accuracy for prompt {prompt_idx+1}. Trying alternative approach...")
        
        # Try using scipy's truncnorm directly
        try:
            from scipy.stats import truncnorm
            
            # Calculate a and b parameters (bounds of truncation in standard deviation units)
            a = (0 - mu_guess) / sigma_guess  # Lower bound
            b = (1 - mu_guess) / sigma_guess  # Upper bound
            
            # Generate samples
            alt_samples = truncnorm.rvs(a, b, loc=mu_guess, scale=sigma_guess, size=n_simulations)
            
            # Check accuracy
            alt_mean = np.mean(alt_samples)
            alt_std = np.std(alt_samples)
            alt_mean_accuracy = abs(alt_mean - target_mean) / target_mean if target_mean != 0 else abs(alt_mean)
            alt_std_accuracy = abs(alt_std - target_std) / target_std if target_std != 0 else abs(alt_std)
            
            print(f"Alternative approach results:")
            print(f"  Alt mean: {alt_mean:.6f}, Rel error: {alt_mean_accuracy:.6f}")
            print(f"  Alt std: {alt_std:.6f}, Rel error: {alt_std_accuracy:.6f}")
            
            # If alternative is better, use it
            if alt_mean_accuracy < mean_accuracy and alt_std_accuracy < std_accuracy:
                print(f"Using alternative approach for prompt {prompt_idx+1}")
                optimized_simulation = alt_samples
                final_sim_mean = alt_mean
                final_sim_std = alt_std
        except Exception as e:
            print(f"Alternative approach failed: {str(e)}")
    
    # Perform KS test between simulated and actual data
    ks_statistic, ks_pvalue = scipy_stats.ks_2samp(values, optimized_simulation)
    
    # Perform Anderson-Darling test between simulated and actual data
    # We'll use a two-sample Anderson-Darling test for this
    try:
        from scipy.stats import anderson_ksamp
        ad_result = anderson_ksamp([values, optimized_simulation])
        ad_statistic = ad_result.statistic
        ad_pvalue = ad_result.pvalue
        ad_significance_level = ad_result.significance_level
        ad_adequate = ad_pvalue > 0.05
    except Exception as e:
        print(f"Anderson-Darling test failed: {str(e)}")
        # Fallback to an approximation if the two-sample test fails
        ad_statistic = np.nan
        ad_pvalue = np.nan
        ad_adequate = False
    
    # Create results dictionary
    results = {
        'Prompt': prompt_idx + 1,
        'Column': column_name,
        'Model Type': 'Truncated Normal with Zero/One Inflation',
        'Underlying Normal Mean': mu_guess,
        'Underlying Normal Std Dev': sigma_guess,
        'Observed Mean': target_mean,
        'Observed Std Dev': target_std,
        'Simulated Mean': final_sim_mean,
        'Simulated Std Dev': final_sim_std,
        'Mean Relative Error': mean_accuracy,
        'Std Relative Error': std_accuracy,
        'Zero Proportion': zero_proportion,
        'One Proportion': one_proportion,
        'Interior Mean': interior_mean,
        'Interior Std Dev': interior_std,
        'KS Statistic': ks_statistic,
        'KS p-value': ks_pvalue,
        'AD Statistic': ad_statistic,
        'AD p-value': ad_pvalue,
        'Model Adequate (KS p>0.05)': ks_pvalue > 0.05,
        'Model Adequate (AD p>0.05)': ad_adequate,
        'Model Adequate (Combined)': (ks_pvalue > 0.05) and ad_adequate
    }
    
    return results, optimized_simulation

# Function to create comparison plot for truncated normal model
def create_truncated_model_plot(data, column_name, prompt_idx, token_options, simulated_data, output_dir, ks_statistic):
    """
    Create a plot comparing actual data to the truncated normal model with zero/one inflation.
    Includes 95% confidence intervals around the QQ plot.
    """
    values = data[column_name].values
    
    # Filter out non-finite values (NaN, inf, -inf)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    
    # Check if we have enough valid data
    if len(values) == 0 or len(simulated_data) == 0:
        print(f"Warning: Insufficient data for truncated model plot for prompt {prompt_idx + 1}, column {column_name}")
        return
    
    # Create the figure with two subplots instead of three
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate basic statistics for actual and simulated data
    actual_mean = np.mean(values)
    actual_std = np.std(values)
    sim_mean = np.mean(simulated_data)
    sim_std = np.std(simulated_data)
    
    # QQ plot of actual vs simulated data
    # Sort both datasets
    sorted_actual = np.sort(values)
    sorted_sim = np.sort(simulated_data)
    
    # If different lengths, interpolate to match
    if len(sorted_actual) != len(sorted_sim):
        # Create evenly spaced quantiles
        quantiles = np.linspace(0, 1, min(len(sorted_actual), len(sorted_sim)))
        # Interpolate both datasets to these quantiles
        from scipy.interpolate import interp1d
        if len(sorted_actual) > len(sorted_sim):
            f_actual = interp1d(np.linspace(0, 1, len(sorted_actual)), sorted_actual)
            actual_matched = f_actual(quantiles)
            sim_matched = sorted_sim
        else:
            f_sim = interp1d(np.linspace(0, 1, len(sorted_sim)), sorted_sim)
            sim_matched = f_sim(quantiles)
            actual_matched = sorted_actual
    else:
        actual_matched = sorted_actual
        sim_matched = sorted_sim
    
    # Calculate the QQ line for reference
    slope, intercept = np.polyfit(sim_matched, actual_matched, 1)
    
    # Generate confidence bands using a parametric approach
    # We'll generate bootstrap samples and transform them to match the regression line
    n = len(sim_matched)
    np.random.seed(42)  # For reproducibility
    n_bootstraps = 1000
    
    # First, get the standard deviation at each point in the QQ plot based on bootstrap
    bootstrap_samples = np.zeros((n_bootstraps, n))
    
    # Generate bootstrap samples
    for i in range(n_bootstraps):
        # Sample with replacement from simulated data
        indices = np.random.randint(0, len(simulated_data), size=n)
        bootstrap_sample = np.sort(simulated_data[indices])
        
        # Interpolate if needed to match the length of sim_matched
        if len(bootstrap_sample) != len(sim_matched):
            f_boot = interp1d(np.linspace(0, 1, len(bootstrap_sample)), bootstrap_sample, 
                              bounds_error=False, fill_value="extrapolate")
            bootstrap_sample = f_boot(np.linspace(0, 1, len(sim_matched)))
        
        bootstrap_samples[i, :] = bootstrap_sample
    
    # Calculate percentiles across simulations for each position
    # These represent the expected distribution of the simulated values
    lower_sim_bounds = np.percentile(bootstrap_samples, 2.5, axis=0)
    upper_sim_bounds = np.percentile(bootstrap_samples, 97.5, axis=0)
    
    # Calculate the deviations of these bounds from the simulated data
    lower_deviations = sim_matched - lower_sim_bounds
    upper_deviations = upper_sim_bounds - sim_matched
    
    # Apply these deviations to the identity line in the observed data space
    # This ensures the CI properly contains the identity line
    lower_ci = actual_matched - (lower_deviations * slope)
    upper_ci = actual_matched + (upper_deviations * slope)
    
    # Make sure we have tight bands for exact correspondences
    # Use a minimum band width to avoid visual artifacts
    min_band_width = 0.01 * (np.max(actual_matched) - np.min(actual_matched))
    band_widths = upper_ci - lower_ci
    for i in range(len(band_widths)):
        if band_widths[i] < min_band_width:
            mid = (upper_ci[i] + lower_ci[i]) / 2
            lower_ci[i] = mid - min_band_width/2
            upper_ci[i] = mid + min_band_width/2
    
    # Plot QQ plot on first subplot
    ax1.scatter(sim_matched, actual_matched, alpha=0.6)
    
    # Add confidence bands
    ax1.fill_between(sim_matched, lower_ci, upper_ci, 
                    color='lightgray', alpha=0.5)
    
    # Add reference line (identity line representing perfect match)
    min_val = min(min(sim_matched), min(actual_matched))
    max_val = max(max(sim_matched), max(actual_matched))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Set labels with increased font size
    if column_name == 'Relative_Prob':
        first_token, second_token = token_options[0], token_options[1]
        ax1.set_xlabel("Truncated Normal Quantiles", fontsize=22)
        ax1.set_ylabel("Actual Data Quantiles", fontsize=22)
    else:
        first_token = token_options[0]
        ax1.set_xlabel("Truncated Normal Quantiles", fontsize=22)
        ax1.set_ylabel("Actual Data Quantiles", fontsize=22)
    
    # Histogram comparison on second subplot
    # Plot histograms of actual and simulated data
    # Calculate histogram bins that work well for both datasets
    # Determine the appropriate range for bins based on the data
    if column_name == 'Weighted Confidence' and actual_mean > 1:  # If confidence data is on a 0-100 scale
        max_bin_value = max(np.max(values), np.max(simulated_data))
        bins = np.linspace(0, max_bin_value, 21)  # 20 bins from 0 to max value
    else:
        bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1 for Relative_Prob or normalized confidence
    
    # Plot actual data (removed label parameter)
    ax2.hist(values, bins=bins, alpha=0.5, color='blue')
    
    # Plot simulated data with the same bins (removed label parameter)
    scaling_factor = len(values) / len(simulated_data)
    counts, _ = np.histogram(simulated_data, bins=bins)
    ax2.hist(simulated_data, bins=bins, alpha=0.5, color='red', 
             weights=np.ones_like(simulated_data) * scaling_factor)
    
    # Set labels with increased font size
    if column_name == 'Relative_Prob':
        ax2.set_xlabel("Relative Probability", fontsize=22)
    else:
        ax2.set_xlabel("Weighted Confidence", fontsize=22)
    
    ax2.set_ylabel("Frequency", fontsize=22)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    if column_name == 'Relative_Prob':
        plt.savefig(output_dir / f'prompt_{prompt_idx + 1}_truncated_model.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_dir / f'prompt_{prompt_idx + 1}_confidence_truncated_model.png', dpi=300, bbox_inches='tight')
    
    plt.close()

# Function to create QQ-plot for visual assessment of normality
def create_qq_plot(data, column_name, prompt_idx, token_options, output_dir):
    """
    Create a QQ-plot to visually assess normality of the distribution.
    Includes 95% confidence intervals around the empirical data points, similar to the truncated model approach.
    """
    # Get values and fit normal distribution
    values = data[column_name].values
    
    # Filter out non-finite values (NaN, inf, -inf)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    
    # Check if we have enough valid data
    if len(values) < 2:
        print(f"Warning: Insufficient data for QQ plot (n={len(values)}) for prompt {prompt_idx + 1}, column {column_name}")
        return
    
    mu, sigma = scipy_stats.norm.fit(values)
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Clear the first subplot
    ax1.clear()
    
    # Get the plotting positions and theoretical quantiles directly
    n = len(values)
    ordered_values = np.sort(values)
    
    # Calculate plotting positions same way scipy does
    # Use the (i - 0.5)/n formula for plotting positions
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = scipy_stats.norm.ppf(plotting_positions)
    
    # Calculate the best fit line for the QQ plot
    # Check if all values are the same (variance is 0)
    if np.var(ordered_values) == 0:
        # If all values are the same, we can't fit a line
        slope = 0
        intercept = ordered_values[0]
    else:
        try:
            slope, intercept = np.polyfit(theoretical_quantiles, ordered_values, 1)
        except np.linalg.LinAlgError:
            # If polyfit fails, use a simple approach
            slope = 0
            intercept = np.mean(ordered_values)
    
    # Generate confidence bands for the empirical data
    # We'll use bootstrap resampling of the empirical data
    np.random.seed(42)  # For reproducibility
    n_bootstraps = 1000
    bootstrap_samples = np.zeros((n_bootstraps, n))
    
    # Generate bootstrap samples from our empirical data
    for i in range(n_bootstraps):
        # Sample with replacement from our actual data
        indices = np.random.randint(0, len(values), size=n)
        sample = values[indices]
        # Sort the bootstrapped sample
        sample = np.sort(sample)
        bootstrap_samples[i, :] = sample
    
    # Calculate percentiles across bootstrap simulations for each position
    lower_ci = np.percentile(bootstrap_samples, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_samples, 97.5, axis=0)
    
    # Make sure we have minimal band width to avoid visual artifacts
    min_band_width = 0.01 * (np.max(ordered_values) - np.min(ordered_values))
    band_widths = upper_ci - lower_ci
    for i in range(len(band_widths)):
        if band_widths[i] < min_band_width:
            mid = (upper_ci[i] + lower_ci[i]) / 2
            lower_ci[i] = mid - min_band_width/2
            upper_ci[i] = mid + min_band_width/2
    
    # Manually create the QQ plot to ensure everything aligns
    ax1.scatter(theoretical_quantiles, ordered_values, alpha=0.6)
    
    # Add confidence intervals to the empirical data
    ax1.fill_between(theoretical_quantiles, lower_ci, upper_ci, 
                    color='lightgray', alpha=0.5)
    
    # Add the QQ line - use a dashed red line to match the truncated plot style
    line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
    line_y = slope * line_x + intercept
    ax1.plot(line_x, line_y, 'r--')  # Changed from 'r-' to 'r--' to match truncated plot
    
    # Set labels for QQ plot with increased font size
    ax1.set_xlabel("Theoretical Quantiles", fontsize=22)
    ax1.set_ylabel("Ordered Values", fontsize=22)
    
    # Plot the histogram with normal curve on the second subplot
    ax2.clear()  # Clear previous data
    sns.histplot(values, kde=False, ax=ax2, bins=20)
    
    # Generate points for normal PDF curve
    x = np.linspace(min(values), max(values), 100)
    y = scipy_stats.norm.pdf(x, mu, sigma)
    y = y * (len(values) * (max(values) - min(values)) / 20)  # Scale to match histogram
    
    # Plot the normal curve
    ax2.plot(x, y, 'r--', linewidth=2)
    
    # Set labels with increased font size
    if column_name == 'Relative_Prob':
        ax2.set_xlabel("Relative Probability", fontsize=22)
    else:
        ax2.set_xlabel("Weighted Confidence", fontsize=22)
    
    ax2.set_ylabel("Frequency", fontsize=22)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    if column_name == 'Relative_Prob':
        plt.savefig(output_dir / f'prompt_{prompt_idx + 1}_qq_plot.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_dir / f'prompt_{prompt_idx + 1}_confidence_qq_plot.png', dpi=300, bbox_inches='tight')
    
    plt.close()

# Function to create histogram for each original prompt
def create_probability_histogram(data, prompt_idx, token_options, output_dir):
    # Filter out non-finite values
    filtered_data = data[np.isfinite(data['Relative_Prob'])]
    
    if len(filtered_data) == 0:
        print(f"Warning: No finite Relative_Prob values for histogram, prompt {prompt_idx + 1}")
        return
    
    plt.figure(figsize=(12, 8))  # Increased height to accommodate legend at bottom
    sns.histplot(data=filtered_data, x='Relative_Prob', bins=10)
    
    # Get the token options
    first_token = token_options[0]
    second_token = token_options[1]
    
    # Remove title and use larger font sizes for labels
    plt.xlabel(f'Relative Probability of "{first_token}" vs "{second_token}"', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    
    # Calculate statistics
    mean_prob = filtered_data['Relative_Prob'].mean()
    lower_percentile = np.percentile(filtered_data['Relative_Prob'], 2.5)
    upper_percentile = np.percentile(filtered_data['Relative_Prob'], 97.5)
    
    # Add vertical lines for mean and 95% confidence interval
    plt.axvline(x=mean_prob, color='r', linestyle='--', 
                label=f'Mean: {mean_prob:.3f}')
    plt.axvline(x=lower_percentile, color='g', linestyle=':', 
                label=f'2.5th percentile: {lower_percentile:.3f}')
    plt.axvline(x=upper_percentile, color='g', linestyle=':', 
                label=f'97.5th percentile: {upper_percentile:.3f}')
    
    # Add shaded region for 95% interval
    plt.axvspan(lower_percentile, upper_percentile, alpha=0.2, color='green')
    
    # Position legend at the bottom outside the plot
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Save the plot with extra padding at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add extra space at bottom for legend
    plt.savefig(output_dir / f'prompt_{prompt_idx + 1}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create histogram for Weighted Confidence
def create_confidence_histogram(data, prompt_idx, token_options, output_dir):
    # Skip if no Weighted Confidence data
    if 'Weighted Confidence' not in data.columns or data['Weighted Confidence'].isna().all():
        print(f"Skipping confidence histogram for prompt {prompt_idx + 1} - no data available")
        return
    
    # Filter out NaN values
    filtered_data = data.dropna(subset=['Weighted Confidence'])
    if len(filtered_data) == 0:
        print(f"Skipping confidence histogram for prompt {prompt_idx + 1} - no valid data after filtering")
        return
    
    plt.figure(figsize=(12, 8))  # Increased height to accommodate legend at bottom
    sns.histplot(data=filtered_data, x='Weighted Confidence', bins=10)
    
    # Get the first token
    first_token = token_options[0]
    
    # Remove title and use larger font sizes for labels
    plt.xlabel(f'Weighted Confidence (0-100) for "{first_token}"', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    
    # Calculate statistics
    mean_conf = filtered_data['Weighted Confidence'].mean()
    lower_percentile = np.percentile(filtered_data['Weighted Confidence'], 2.5)
    upper_percentile = np.percentile(filtered_data['Weighted Confidence'], 97.5)
    
    # Add vertical lines for mean and 95% confidence interval
    plt.axvline(x=mean_conf, color='r', linestyle='--', 
                label=f'Mean: {mean_conf:.1f}')
    plt.axvline(x=lower_percentile, color='g', linestyle=':', 
                label=f'2.5th percentile: {lower_percentile:.1f}')
    plt.axvline(x=upper_percentile, color='g', linestyle=':', 
                label=f'97.5th percentile: {upper_percentile:.1f}')
    
    # Add shaded region for 95% interval
    plt.axvspan(lower_percentile, upper_percentile, alpha=0.2, color='green')
    
    # Add a reference line at 50 (neutral confidence)
    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    
    # Position legend at the bottom outside the plot
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Save the plot with extra padding at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add extra space at bottom for legend
    plt.savefig(output_dir / f'prompt_{prompt_idx + 1}_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create LaTeX table content
def create_latex_table(data, prompt_idx, prompt_info, output_dir):
    # Get a descriptive name for the prompt based on its content
    prompt_descriptions = [
        "Insurance Policy Water Damage Exclusion",
        "Prenuptial Agreement Petition Filing Date",
        "Contract Term Affiliate Interpretation",
        "Construction Payment Terms Interpretation",
        "Insurance Policy Burglary Coverage"
    ]
    
    # Get the token options
    token_options = prompt_info[2]
    first_token = token_options[0]
    second_token = token_options[1]
    
    # Use the appropriate description if available, otherwise use a generic one
    description = prompt_descriptions[prompt_idx] if prompt_idx < len(prompt_descriptions) else f"Prompt {prompt_idx + 1}"
    
    # Check if we have Weighted Confidence data
    has_confidence_data = 'Weighted Confidence' in data.columns and not data['Weighted Confidence'].isna().all()
    
    latex_content = []
    
    # Add the subsection header with the original prompt
    latex_content.append(f"\\subsection*{{Prompt {prompt_idx + 1}: {description}}}")
    latex_content.append("")
    latex_content.append(f"\\textbf{{Original Prompt:}} {prompt_info[0]}")
    latex_content.append("")
    
    # Create the next-token distribution table
    latex_content.append("\\subsubsection*{Next-Token Distribution Table}")
    latex_content.append("")
    latex_content.append("\\begin{longtable}{p{0.65\\textwidth}cc}")
    latex_content.append(f"\\caption{{Representative Relative Probabilities for {description}: \"{first_token}\" vs \"{second_token}\" (Prompt {prompt_idx + 1})}} \\\\")
    latex_content.append("\\hline")
    latex_content.append("Prompt Variation & \\makecell{Relative\\\\Probability} & Percentile \\\\")
    latex_content.append("\\hline")
    latex_content.append("\\endhead")
    latex_content.append("\\hline")
    latex_content.append("\\endfoot")
    
    # Select 20 representative prompts from different percentile chunks
    # Filter out non-finite values first
    finite_data = data[np.isfinite(data['Relative_Prob'])]
    
    if len(finite_data) == 0:
        latex_content.append("No valid data available for this prompt. & - & - \\\\")
        latex_content.append("\\end{longtable}")
        latex_content.append("")
        return '\n'.join(latex_content)
    
    # Sort data by Relative_Prob
    sorted_data = finite_data.sort_values('Relative_Prob')
    
    # Create 20 percentile chunks and select one random sample from each
    num_chunks = 20
    chunk_size = len(sorted_data) // num_chunks
    
    # Handle cases where there are fewer than 20 samples
    if chunk_size == 0:
        selected_rows = sorted_data
    else:
        selected_rows = pd.DataFrame()
        for i in range(num_chunks):
            start_idx = i * chunk_size
            # For the last chunk, include all remaining rows
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(sorted_data)
            
            if start_idx < end_idx:  # Ensure the chunk has at least one row
                chunk = sorted_data.iloc[start_idx:end_idx]
                # Randomly select one row from the chunk
                selected_row = chunk.sample(n=1)
                selected_rows = pd.concat([selected_rows, selected_row])
    
    # Add the selected variations to the table
    for _, row in selected_rows.iterrows():
        escaped_prompt = row['Full Rephrased Prompt'].replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
        prob = float(row['Relative_Prob'])  # Convert to scalar float
        # Calculate percentile for this specific value rather than looking it up in the Series
        percentile = 100 * (sorted_data['Relative_Prob'] <= prob).mean()
        
        latex_content.append(f"{escaped_prompt} & {prob:.3f} & {percentile:.1f}\\% \\\\")
    
    latex_content.append("\\end{longtable}")
    latex_content.append("")
    
    # Create the confidence estimates table if data is available
    if has_confidence_data:
        latex_content.append("\\subsubsection*{Confidence Estimates Table}")
        latex_content.append("")
        latex_content.append("\\begin{longtable}{p{0.65\\textwidth}cc}")
        latex_content.append(f"\\caption{{Representative Weighted Confidence for {description}: \"{first_token}\" (Prompt {prompt_idx + 1})}} \\\\")
        latex_content.append("\\hline")
        latex_content.append("Prompt Variation & \\makecell{Weighted\\\\Confidence} & Percentile \\\\")
        latex_content.append("\\hline")
        latex_content.append("\\endhead")
        latex_content.append("\\hline")
        latex_content.append("\\endfoot")
        
        # Filter out rows with NaN confidence values
        filtered_data = data.dropna(subset=['Weighted Confidence'])
        
        if len(filtered_data) > 0:
            # Sort by confidence
            sorted_conf_data = filtered_data.sort_values('Weighted Confidence')
            
            # Create chunks for confidence data
            num_chunks = min(20, len(sorted_conf_data))
            chunk_size = len(sorted_conf_data) // num_chunks
            
            # Handle cases where there are fewer than 20 samples
            if chunk_size == 0:
                selected_conf_rows = sorted_conf_data
            else:
                selected_conf_rows = pd.DataFrame()
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    # For the last chunk, include all remaining rows
                    end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(sorted_conf_data)
                    
                    if start_idx < end_idx:  # Ensure the chunk has at least one row
                        chunk = sorted_conf_data.iloc[start_idx:end_idx]
                        # Randomly select one row from the chunk
                        selected_row = chunk.sample(n=1)
                        selected_conf_rows = pd.concat([selected_conf_rows, selected_row])
            
            # Add the selected variations to the confidence table
            for _, row in selected_conf_rows.iterrows():
                # Use Full Confidence Prompt instead of Full Rephrased Prompt for confidence tables
                escaped_prompt = row['Full Confidence Prompt'].replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
                conf = float(row['Weighted Confidence'])  # Convert to scalar float
                # Calculate percentile for this specific value rather than looking it up in the Series
                percentile = 100 * (sorted_conf_data['Weighted Confidence'] <= conf).mean()
                
                latex_content.append(f"{escaped_prompt} & {conf:.1f} & {percentile:.1f}\\% \\\\")
        else:
            latex_content.append("No confidence data available for this prompt. & - & - \\\\")
        
        latex_content.append("\\end{longtable}")
        latex_content.append("")
    
    return '\n'.join(latex_content)

def create_standalone_latex_document(all_tables):
    """Create a complete standalone LaTeX document with proper formatting."""
    
    # LaTeX preamble matching main.tex style
    preamble = r"""\documentclass[12pt]{article}
\usepackage{amsfonts}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage[margin=1.25in]{geometry}
\usepackage{natbib}
\usepackage{longtable}
\usepackage{subcaption}
\usepackage{graphicx}
\usepackage{makecell}
\usepackage{float}
\usepackage{amsmath}
\usepackage{setspace}
\usepackage{comment}
\usepackage[font=normal,labelfont=bf,skip=6pt]{caption}

% Add more space between paragraphs
\setlength{\parskip}{0.5em}

\title{Prompt Perturbation Analysis Appendix}
\author{}
\date{\today}

\begin{document}
\maketitle

\section*{Prompt Perturbation Analysis}

This appendix presents the detailed results of the prompt perturbation analysis discussed in \textit{Off-the-Shelf Large Language Models Are Unreliable Judges}. For each of the five legal interpretation prompts, I first show the original prompt in plain text, followed by a table containing 20 representative prompt variations selected from different percentile ranges of the distribution. Each table displays the rephrasings along with their relative probability (the probability of the first token divided by the sum of probabilities for both possible answer tokens) and the percentile rank of that probability within the distribution. The representative prompts are systematically sampled across the full distribution to illustrate the range of model responses to inputs generated from the same underlying set of facts.

"""
    
    # Document footer
    footer = r"""
\end{document}"""
    
    # Combine all parts
    full_document = preamble + '\n'.join(all_tables) + footer
    
    return full_document

# Function to create a combined visualization of all prompts
def create_combined_visualization(df, prompts, output_dir):
    plt.figure(figsize=(14, 10))  # Increased height to accommodate legend at bottom
    
    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()
    
    # Create a new column for prompt number (for plotting)
    df['Prompt_Number'] = df['Original Main Part'].apply(lambda x: list(unique_prompts).index(x) + 1)
    
    # Set up the plot
    ax = plt.subplot(111)
    
    # Define colors for each prompt
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot violin plots and jittered points for each prompt
    for idx, prompt in enumerate(unique_prompts):
        prompt_data = df[df['Original Main Part'] == prompt]
        
        # Calculate statistics
        mean_prob = prompt_data['Relative_Prob'].mean()
        lower_percentile = np.percentile(prompt_data['Relative_Prob'], 2.5)
        upper_percentile = np.percentile(prompt_data['Relative_Prob'], 97.5)
        
        # Get token options
        token_options = prompts[idx][2]
        first_token = token_options[0]
        second_token = token_options[1]
        
        # Add violin plot (with lower alpha to not obscure points)
        violin_parts = ax.violinplot([prompt_data['Relative_Prob']], [idx + 1], 
                                    widths=0.3, showmeans=False, showmedians=False, showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[idx % len(colors)])
            pc.set_edgecolor('none')
            pc.set_alpha(0.3)
        
        # Add jittered points
        x_jittered = np.random.normal(idx + 1, 0.08, size=len(prompt_data))
        plt.scatter(x_jittered, prompt_data['Relative_Prob'], alpha=0.4, s=30, 
                   color=colors[idx % len(colors)], label=f'Prompt {idx + 1}' if idx == 0 else "")
        
        # Add mean point
        plt.scatter(idx + 1, mean_prob, color='black', s=80, zorder=5)
        
        # Add error bars for 95% CI
        plt.plot([idx + 1, idx + 1], [lower_percentile, upper_percentile], 
                color='black', linewidth=2, zorder=4)
        
        # Add caps to the error bars
        cap_width = 0.1
        plt.plot([idx + 1 - cap_width, idx + 1 + cap_width], 
                [lower_percentile, lower_percentile], color='black', linewidth=2, zorder=4)
        plt.plot([idx + 1 - cap_width, idx + 1 + cap_width], 
                [upper_percentile, upper_percentile], color='black', linewidth=2, zorder=4)
    
    # Add a horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Set the x-ticks and labels - MODIFIED: only show prompt numbers
    plt.xticks(range(1, len(unique_prompts) + 1), 
              [f"{i+1}" for i in range(len(unique_prompts))], fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add labels but no title
    plt.ylabel('Relative Probability of First Token', fontsize=20)
    
    # Create custom legend elements
    custom_legend = []
    for idx, prompt in enumerate(unique_prompts):
        token_options = prompts[idx][2]
        first_token = token_options[0]
        second_token = token_options[1]
        custom_legend.append(plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=colors[idx % len(colors)], markersize=10, 
                                       label=f"Prompt {idx+1}: '{first_token}' vs '{second_token}'"))
    
    # Add the legend at the bottom of the plot
    plt.legend(handles=custom_legend, fontsize=16, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=1)
    
    # Adjust layout and save with extra space at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add extra space at bottom for legend
    plt.savefig(output_dir / 'combined_prompts_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to create a combined visualization for Weighted Confidence
def create_combined_confidence_visualization(df, prompts, output_dir):
    # Check if we have Weighted Confidence data
    if 'Weighted Confidence' not in df.columns or df['Weighted Confidence'].isna().all():
        print("Skipping combined confidence visualization - no data available")
        return
    
    plt.figure(figsize=(14, 10))  # Increased height to accommodate legend at bottom
    
    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()
    
    # Create a new column for prompt number (for plotting)
    df['Prompt_Number'] = df['Original Main Part'].apply(lambda x: list(unique_prompts).index(x) + 1)
    
    # Set up the plot
    ax = plt.subplot(111)
    
    # Define colors for each prompt
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot violin plots and jittered points for each prompt
    for idx, prompt in enumerate(unique_prompts):
        prompt_data = df[df['Original Main Part'] == prompt].dropna(subset=['Weighted Confidence'])
        
        if len(prompt_data) == 0:
            print(f"Skipping prompt {idx+1} in combined confidence visualization - no valid data")
            continue
        
        # Calculate statistics
        mean_conf = prompt_data['Weighted Confidence'].mean()
        lower_percentile = np.percentile(prompt_data['Weighted Confidence'], 2.5)
        upper_percentile = np.percentile(prompt_data['Weighted Confidence'], 97.5)
        
        # Get token options
        token_options = prompts[idx][2]
        first_token = token_options[0]
        
        # Add violin plot (with lower alpha to not obscure points)
        violin_parts = ax.violinplot([prompt_data['Weighted Confidence']], [idx + 1], 
                                    widths=0.3, showmeans=False, showmedians=False, showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[idx % len(colors)])
            pc.set_edgecolor('none')
            pc.set_alpha(0.3)
        
        # Add jittered points
        x_jittered = np.random.normal(idx + 1, 0.08, size=len(prompt_data))
        plt.scatter(x_jittered, prompt_data['Weighted Confidence'], alpha=0.4, s=30, 
                   color=colors[idx % len(colors)], label=f'Prompt {idx + 1}' if idx == 0 else "")
        
        # Add mean point
        plt.scatter(idx + 1, mean_conf, color='black', s=80, zorder=5)
        
        # Add error bars for 95% CI
        plt.plot([idx + 1, idx + 1], [lower_percentile, upper_percentile], 
                color='black', linewidth=2, zorder=4)
        
        # Add caps to the error bars
        cap_width = 0.1
        plt.plot([idx + 1 - cap_width, idx + 1 + cap_width], 
                [lower_percentile, lower_percentile], color='black', linewidth=2, zorder=4)
        plt.plot([idx + 1 - cap_width, idx + 1 + cap_width], 
                [upper_percentile, upper_percentile], color='black', linewidth=2, zorder=4)
    
    # Add a horizontal line at 50 for reference (neutral confidence)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # Set the x-ticks and labels - MODIFIED: only show prompt numbers
    plt.xticks(range(1, len(unique_prompts) + 1), 
              [f"{i+1}" for i in range(len(unique_prompts))], fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add labels but no title
    plt.ylabel('Weighted Confidence (0-100)', fontsize=20)
    
    # Create custom legend elements
    custom_legend = []
    for idx, prompt in enumerate(unique_prompts):
        token_options = prompts[idx][2]
        first_token = token_options[0]
        custom_legend.append(plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=colors[idx % len(colors)], markersize=10, 
                                       label=f"Prompt {idx+1}: Confidence for '{first_token}'"))
    
    # Add the legend at the bottom of the plot
    plt.legend(handles=custom_legend, fontsize=16, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=1)
    
    # Adjust layout and save with extra space at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add extra space at bottom for legend
    plt.savefig(output_dir / 'combined_confidence_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to calculate Cohen's kappa for binary decisions across prompts
def calculate_cohens_kappa(df):
    """
    Calculate Cohen's kappa to measure agreement between evaluations based on same prompt
    compared to random evaluations across different prompts.
    
    The function:
    1. Converts relative probabilities to binary decisions (first token vs second token)
    2. Calculates observed agreement for same prompt pairs
    3. Calculates expected agreement by chance (using all possible prompt pairs)
    4. Computes Cohen's kappa from these values
    """
    print("\nCalculating Cohen's kappa for agreement between evaluations...")
    
    # Filter out non-finite values first
    finite_df = df[np.isfinite(df['Relative_Prob'])].copy()
    
    if len(finite_df) == 0:
        print("Warning: No finite relative probability values for Cohen's kappa calculation")
        return np.nan, np.nan, np.nan
    
    # Convert relative probabilities to binary decisions (0 or 1)
    # If Relative_Prob > 0.5, choose first token (1), otherwise choose second token (0)
    finite_df['Binary_Decision'] = (finite_df['Relative_Prob'] > 0.5).astype(int)
    
    # Group by original prompt
    prompt_groups = finite_df.groupby('Original Main Part')
    
    # Calculate observed agreement (agreement within same prompt)
    same_prompt_agreement_count = 0
    same_prompt_pair_count = 0
    
    # For each prompt, calculate agreement between all pairs of evaluations
    for prompt, group in prompt_groups:
        decisions = group['Binary_Decision'].values
        n = len(decisions)
        
        if n <= 1:
            continue  # Skip prompts with only one evaluation
            
        # Count pairs that agree
        for i in range(n):
            for j in range(i+1, n):
                same_prompt_pair_count += 1
                if decisions[i] == decisions[j]:
                    same_prompt_agreement_count += 1
    
    # Calculate observed agreement rate
    if same_prompt_pair_count > 0:
        observed_agreement = same_prompt_agreement_count / same_prompt_pair_count
    else:
        observed_agreement = 0
        print("Warning: No valid pairs found for same-prompt agreement calculation")
    
    # Calculate expected agreement by chance (across all prompts)
    all_decisions = finite_df['Binary_Decision'].values
    p1 = np.mean(all_decisions)  # Proportion of '1' decisions
    p0 = 1 - p1  # Proportion of '0' decisions
    
    # Expected agreement by chance: probability both are 1 or both are 0
    expected_agreement = (p1 * p1) + (p0 * p0)
    
    # Calculate Cohen's kappa
    if expected_agreement < 1:  # Avoid division by zero
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    else:
        kappa = 1.0
    
    # Print detailed results
    print(f"\nCohen's Kappa Analysis Results:")
    print(f"Total evaluations: {len(finite_df)} (after filtering non-finite values)")
    print(f"Number of unique prompts: {len(prompt_groups)}")
    print(f"Same-prompt pair count: {same_prompt_pair_count}")
    print(f"Same-prompt agreements: {same_prompt_agreement_count}")
    print(f"Observed agreement rate: {observed_agreement:.4f}")
    print(f"Expected agreement rate: {expected_agreement:.4f}")
    print(f"Cohen's kappa: {kappa:.4f}")
    
    # Interpret kappa value
    if kappa < 0:
        interpretation = "Poor agreement (worse than chance)"
    elif kappa < 0.2:
        interpretation = "Slight agreement"
    elif kappa < 0.4:
        interpretation = "Fair agreement"
    elif kappa < 0.6:
        interpretation = "Moderate agreement"
    elif kappa < 0.8:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"Interpretation: {interpretation}")
    
    return kappa, observed_agreement, expected_agreement

# Function to check compliance with output instructions
def check_output_compliance(df, prompts, output_dir):
    """
    Check compliance with output instructions by analyzing:
    1. Whether the most probable first token matches expected tokens
    2. Whether subsequent tokens match expected patterns (when first token is compliant)
    
    Returns a DataFrame with compliance statistics for each prompt.
    """
    import json
    import ast
    
    print("\n" + "="*60)
    print("CHECKING OUTPUT INSTRUCTION COMPLIANCE")
    print("="*60)
    
    # Define expected tokens for each prompt
    expected_tokens = [
        # Prompt 1: Insurance policy water damage
        {
            'first_tokens': ['Covered', 'Not'],
            'full_responses': {
                'Covered': ['Covered'],
                'Not': ['Not Covered', 'Not covered']
            }
        },
        # Prompt 2: Prenuptial agreement petition
        {
            'first_tokens': ['First', 'Ultimate'],
            'full_responses': {
                'First': ['First Petition', 'First petition'],
                'Ultimate': ['Ultimate Petition', 'Ultimate petition']
            }
        },
        # Prompt 3: Contract term affiliates
        {
            'first_tokens': ['Existing', 'Future'],
            'full_responses': {
                'Existing': ['Existing Affiliates', 'Existing affiliates'],
                'Future': ['Future Affiliates', 'Future affiliates']
            }
        },
        # Prompt 4: Construction payment terms
        {
            'first_tokens': ['Monthly', 'Payment'],
            'full_responses': {
                'Monthly': ['Monthly Installment Payments', 'Monthly installment payments', 'Monthly Installment Payment'],
                'Payment': ['Payment Upon Completion', 'Payment upon completion', 'Payment Upon']
            }
        },
        # Prompt 5: Insurance policy burglary
        {
            'first_tokens': ['Covered', 'Not'],
            'full_responses': {
                'Covered': ['Covered'],
                'Not': ['Not Covered', 'Not covered']
            }
        }
    ]
    
    # Check if we have log probabilities column
    has_log_probs = 'Log Probabilities' in df.columns
    
    if not has_log_probs:
        print("Warning: Log Probabilities column not found in data. Cannot perform compliance check.")
        print("Available columns:", df.columns.tolist())
        return pd.DataFrame()
    
    # Initialize results storage
    compliance_results = []
    
    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()
    
    for idx, original_prompt in enumerate(unique_prompts):
        if idx >= len(expected_tokens):
            print(f"Warning: No expected tokens defined for prompt {idx + 1}")
            continue
            
        prompt_data = df[df['Original Main Part'] == original_prompt].copy()
        prompt_info = prompts[idx]
        token_options = prompt_info[2]
        
        print(f"\nPrompt {idx + 1}: Checking compliance...")
        print(f"  Expected first tokens: {expected_tokens[idx]['first_tokens']}")
        
        # Filter out rows with non-finite relative probabilities
        valid_data = prompt_data[np.isfinite(prompt_data['Relative_Prob'])].copy()
        total_samples = len(valid_data)
        
        if total_samples == 0:
            print(f"  No valid data for prompt {idx + 1}")
            continue
        
        # Check first token compliance and subsequent token compliance
        first_token_compliant = 0
        first_token_non_compliant = 0
        subsequent_compliant = 0
        subsequent_non_compliant = 0
        compliant_first_samples = 0
        
        # Track unique non-compliant examples
        non_compliant_first_examples = set()
        non_compliant_full_examples = set()
        
        for _, row in valid_data.iterrows():
            # Parse the log probabilities to get the actual first token
            try:
                log_probs = row['Log Probabilities']
                
                # Handle string representation of dict
                if isinstance(log_probs, str):
                    # Try to parse as JSON first
                    try:
                        log_probs = json.loads(log_probs)
                    except json.JSONDecodeError:
                        # If JSON fails, try ast.literal_eval
                        try:
                            log_probs = ast.literal_eval(log_probs)
                        except (ValueError, SyntaxError):
                            continue
                
                # Extract the first token from the content
                if 'content' in log_probs and len(log_probs['content']) > 0:
                    first_token_info = log_probs['content'][0]
                    actual_first_token = first_token_info.get('token', '')
                    
                    # Build full response from all tokens
                    full_response_tokens = []
                    for token_info in log_probs['content']:
                        token_text = token_info.get('token', '')
                        full_response_tokens.append(token_text)
                    full_response = ''.join(full_response_tokens).strip()
                    
                else:
                    continue
                    
            except Exception as e:
                continue
            
            # Check first token compliance
            is_first_compliant = False
            expected_first_token = None
            
            for expected_first in expected_tokens[idx]['first_tokens']:
                # Check if the actual first token matches or starts with the expected token
                if actual_first_token == expected_first or actual_first_token.startswith(expected_first):
                    is_first_compliant = True
                    expected_first_token = expected_first
                    break
            
            if is_first_compliant:
                first_token_compliant += 1
                compliant_first_samples += 1
                
                # Check subsequent token compliance (full response)
                expected_responses = expected_tokens[idx]['full_responses'].get(expected_first_token, [])
                
                is_full_compliant = False
                for expected_full in expected_responses:
                    # Check exact match or if response matches expected pattern
                    # Remove spaces for comparison since tokenization might split differently
                    normalized_response = full_response.replace(' ', '')
                    normalized_expected = expected_full.replace(' ', '')
                    
                    if (full_response == expected_full or 
                        normalized_response == normalized_expected or
                        normalized_response.startswith(normalized_expected)):
                        is_full_compliant = True
                        break
                
                if is_full_compliant:
                    subsequent_compliant += 1
                else:
                    subsequent_non_compliant += 1
                    # Track non-compliant full responses
                    if len(non_compliant_full_examples) < 5:
                        non_compliant_full_examples.add(f"'{full_response}'")
            else:
                first_token_non_compliant += 1
                # Track non-compliant first tokens
                if len(non_compliant_first_examples) < 5:
                    non_compliant_first_examples.add(f"'{actual_first_token}'")
        
        first_token_compliance_rate = (first_token_compliant / total_samples) * 100 if total_samples > 0 else 0
        first_token_non_compliance_rate = (first_token_non_compliant / total_samples) * 100 if total_samples > 0 else 0
        
        # Calculate conditional subsequent compliance rates
        subsequent_compliance_rate = None
        subsequent_non_compliance_rate = None
        
        if compliant_first_samples > 0:
            subsequent_compliance_rate = (subsequent_compliant / compliant_first_samples) * 100
            subsequent_non_compliance_rate = (subsequent_non_compliant / compliant_first_samples) * 100
        
        # Store results
        result = {
            'Prompt': idx + 1,
            'Expected_First_Tokens': ', '.join(expected_tokens[idx]['first_tokens']),
            'Total_Samples': total_samples,
            'First_Token_Compliant': first_token_compliant,
            'First_Token_Non_Compliant': first_token_non_compliant,
            'First_Token_Compliance_Rate': first_token_compliance_rate,
            'First_Token_Non_Compliance_Rate': first_token_non_compliance_rate
        }
        
        if subsequent_compliance_rate is not None:
            result.update({
                'Conditional_Subsequent_Compliant': subsequent_compliant,
                'Conditional_Subsequent_Non_Compliant': subsequent_non_compliant,
                'Conditional_Subsequent_Compliance_Rate': subsequent_compliance_rate,
                'Conditional_Subsequent_Non_Compliance_Rate': subsequent_non_compliance_rate
            })
        
        compliance_results.append(result)
        
        # Print summary for this prompt
        print(f"  First token non-compliance rate: {first_token_non_compliance_rate:.3f}%")
        if non_compliant_first_examples:
            print(f"    Non-compliant first token examples: {', '.join(sorted(non_compliant_first_examples))}")
        
        if subsequent_non_compliance_rate is not None:
            print(f"  Conditional subsequent non-compliance rate: {subsequent_non_compliance_rate:.3f}%")
            if non_compliant_full_examples:
                print(f"    Non-compliant full response examples: {', '.join(sorted(non_compliant_full_examples))}")
    
    # Create DataFrame with results
    compliance_df = pd.DataFrame(compliance_results)
    
    # Calculate overall statistics
    if len(compliance_df) > 0:
        overall_first_non_compliance = (
            compliance_df['First_Token_Non_Compliant'].sum() / 
            compliance_df['Total_Samples'].sum() * 100
        )
        
        print(f"\n" + "="*60)
        print("OVERALL COMPLIANCE STATISTICS")
        print("="*60)
        print(f"Overall first token non-compliance rate: {overall_first_non_compliance:.3f}%")
        
        if 'Conditional_Subsequent_Non_Compliance_Rate' in compliance_df.columns:
            # Calculate weighted average of conditional subsequent non-compliance
            valid_prompts = compliance_df[compliance_df['Conditional_Subsequent_Non_Compliance_Rate'].notna()]
            if len(valid_prompts) > 0:
                weights = valid_prompts['First_Token_Compliant']
                rates = valid_prompts['Conditional_Subsequent_Non_Compliance_Rate']
                if weights.sum() > 0:
                    overall_subsequent_non_compliance = (weights * rates).sum() / weights.sum()
                    print(f"Overall conditional subsequent non-compliance rate: {overall_subsequent_non_compliance:.3f}%")
        
        # Save compliance results
        compliance_df.to_csv(output_dir / 'output_compliance_results.csv', index=False)
        print(f"\nCompliance results saved to: {output_dir / 'output_compliance_results.csv'}")
        
        # Create a summary table for LaTeX
        latex_summary = create_compliance_latex_table(compliance_df)
        with open(output_dir / 'compliance_summary.tex', 'w', encoding='utf-8') as f:
            f.write(latex_summary)
        print(f"LaTeX compliance summary saved to: {output_dir / 'compliance_summary.tex'}")
    
    return compliance_df

def create_compliance_latex_table(compliance_df):
    """Create a LaTeX table summarizing compliance results."""
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Output Instruction Compliance Analysis}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\hline")
    latex.append("Prompt & \\makecell{First Token\\\\Non-Compliance (\\%)} & \\makecell{Conditional Subsequent\\\\Non-Compliance (\\%)} & \\makecell{Total\\\\Samples} \\\\")
    latex.append("\\hline")
    
    for _, row in compliance_df.iterrows():
        prompt_num = row['Prompt']
        first_non_comp = row['First_Token_Non_Compliance_Rate']
        total_samples = row['Total_Samples']
        
        if 'Conditional_Subsequent_Non_Compliance_Rate' in row and pd.notna(row['Conditional_Subsequent_Non_Compliance_Rate']):
            subsequent_non_comp = row['Conditional_Subsequent_Non_Compliance_Rate']
            latex.append(f"{prompt_num} & {first_non_comp:.3f} & {subsequent_non_comp:.3f} & {total_samples} \\\\")
        else:
            latex.append(f"{prompt_num} & {first_non_comp:.3f} & N/A & {total_samples} \\\\")
    
    latex.append("\\hline")
    
    # Add overall statistics
    overall_first_non_comp = (compliance_df['First_Token_Non_Compliant'].sum() / 
                             compliance_df['Total_Samples'].sum() * 100)
    total_all = compliance_df['Total_Samples'].sum()
    
    if 'Conditional_Subsequent_Non_Compliance_Rate' in compliance_df.columns:
        valid_prompts = compliance_df[compliance_df['Conditional_Subsequent_Non_Compliance_Rate'].notna()]
        if len(valid_prompts) > 0:
            weights = valid_prompts['First_Token_Compliant']
            rates = valid_prompts['Conditional_Subsequent_Non_Compliance_Rate']
            if weights.sum() > 0:
                overall_subsequent_non_comp = (weights * rates).sum() / weights.sum()
                latex.append(f"\\textbf{{Overall}} & \\textbf{{{overall_first_non_comp:.3f}}} & \\textbf{{{overall_subsequent_non_comp:.3f}}} & \\textbf{{{total_all}}} \\\\")
            else:
                latex.append(f"\\textbf{{Overall}} & \\textbf{{{overall_first_non_comp:.3f}}} & N/A & \\textbf{{{total_all}}} \\\\")
    else:
        latex.append(f"\\textbf{{Overall}} & \\textbf{{{overall_first_non_comp:.3f}}} & N/A & \\textbf{{{total_all}}} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return '\n'.join(latex)

def check_confidence_compliance(df, prompts, output_dir):
    """
    Check compliance with confidence output instructions by analyzing:
    Whether the confidence responses are integers as expected (not floats, text, etc.)
    
    Returns a DataFrame with confidence compliance statistics for each prompt.
    """
    print("\n" + "="*60)
    print("CHECKING CONFIDENCE OUTPUT COMPLIANCE")
    print("="*60)
    
    # Check if we have the confidence response column
    if 'Model Confidence Response' not in df.columns:
        print("Warning: Model Confidence Response column not found. Cannot check confidence compliance.")
        print("Available columns:", df.columns.tolist())
        return pd.DataFrame()
    
    # Initialize results storage
    confidence_compliance_results = []
    
    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()
    
    for idx, original_prompt in enumerate(unique_prompts):
        if idx >= len(prompts):
            print(f"Warning: No prompt info for prompt {idx + 1}")
            continue
            
        prompt_data = df[df['Original Main Part'] == original_prompt].copy()
        
        print(f"\nPrompt {idx + 1}: Checking confidence compliance...")
        
        # Filter out rows with confidence responses
        # Some rows might not have confidence responses if the model didn't provide them
        valid_data = prompt_data[prompt_data['Model Confidence Response'].notna()].copy()
        total_samples = len(valid_data)
        
        if total_samples == 0:
            print(f"  No confidence responses for prompt {idx + 1}")
            continue
        
        # Check confidence compliance
        compliant = 0
        non_compliant = 0
        non_compliant_examples = set()
        
        # Track types of non-compliance
        non_compliance_types = {
            'float': 0,
            'text': 0,
            'out_of_range': 0,
            'other': 0
        }
        
        for _, row in valid_data.iterrows():
            confidence_response = row['Model Confidence Response']
            
            try:
                # Convert to string and strip whitespace
                conf_str = str(confidence_response).strip()
                
                # Check if it's a valid integer
                is_compliant = False
                
                # Try to parse as integer
                try:
                    conf_value = int(conf_str)
                    # Check if it's in valid range (0-100)
                    if 0 <= conf_value <= 100:
                        is_compliant = True
                        compliant += 1
                    else:
                        # Integer but out of range
                        non_compliant += 1
                        non_compliance_types['out_of_range'] += 1
                        if len(non_compliant_examples) < 5:
                            non_compliant_examples.add(f"'{conf_str}' (out of range)")
                except ValueError:
                    # Not an integer
                    non_compliant += 1
                    
                    # Check if it's a float
                    try:
                        float_val = float(conf_str)
                        # It's a float, not an integer
                        non_compliance_types['float'] += 1
                        if len(non_compliant_examples) < 5:
                            non_compliant_examples.add(f"'{conf_str}' (float)")
                    except ValueError:
                        # It's text or other non-numeric
                        if any(c.isalpha() for c in conf_str):
                            non_compliance_types['text'] += 1
                            if len(non_compliant_examples) < 5:
                                non_compliant_examples.add(f"'{conf_str}' (text)")
                        else:
                            non_compliance_types['other'] += 1
                            if len(non_compliant_examples) < 5:
                                non_compliant_examples.add(f"'{conf_str}' (other)")
                
            except Exception as e:
                # Error processing this response
                non_compliant += 1
                non_compliance_types['other'] += 1
                continue
        
        # Calculate rates
        compliance_rate = (compliant / total_samples) * 100 if total_samples > 0 else 0
        non_compliance_rate = (non_compliant / total_samples) * 100 if total_samples > 0 else 0
        
        # Store results
        result = {
            'Prompt': idx + 1,
            'Total_Confidence_Samples': total_samples,
            'Confidence_Compliant': compliant,
            'Confidence_Non_Compliant': non_compliant,
            'Confidence_Compliance_Rate': compliance_rate,
            'Confidence_Non_Compliance_Rate': non_compliance_rate,
            'Float_Errors': non_compliance_types['float'],
            'Text_Errors': non_compliance_types['text'],
            'Out_Of_Range_Errors': non_compliance_types['out_of_range'],
            'Other_Errors': non_compliance_types['other']
        }
        
        confidence_compliance_results.append(result)
        
        # Print summary for this prompt
        print(f"  Confidence non-compliance rate: {non_compliance_rate:.3f}%")
        if non_compliant_examples:
            print(f"    Non-compliant examples: {', '.join(sorted(non_compliant_examples))}")
        if non_compliant > 0:
            print(f"    Error breakdown: Float={non_compliance_types['float']}, "
                  f"Text={non_compliance_types['text']}, "
                  f"Out-of-range={non_compliance_types['out_of_range']}, "
                  f"Other={non_compliance_types['other']}")
    
    # Create DataFrame with results
    confidence_df = pd.DataFrame(confidence_compliance_results)
    
    # Calculate overall statistics
    if len(confidence_df) > 0:
        overall_non_compliance = (
            confidence_df['Confidence_Non_Compliant'].sum() / 
            confidence_df['Total_Confidence_Samples'].sum() * 100
        )
        
        print(f"\n" + "="*60)
        print("OVERALL CONFIDENCE COMPLIANCE STATISTICS")
        print("="*60)
        print(f"Overall confidence non-compliance rate: {overall_non_compliance:.3f}%")
        
        # Show breakdown of error types
        total_errors = confidence_df['Confidence_Non_Compliant'].sum()
        if total_errors > 0:
            float_pct = (confidence_df['Float_Errors'].sum() / total_errors) * 100
            text_pct = (confidence_df['Text_Errors'].sum() / total_errors) * 100
            range_pct = (confidence_df['Out_Of_Range_Errors'].sum() / total_errors) * 100
            other_pct = (confidence_df['Other_Errors'].sum() / total_errors) * 100
            
            print(f"Error type breakdown:")
            print(f"  Float values: {float_pct:.3f}%")
            print(f"  Text responses: {text_pct:.3f}%")
            print(f"  Out of range [0-100]: {range_pct:.3f}%")
            print(f"  Other errors: {other_pct:.3f}%")
        
        # Save confidence compliance results
        confidence_df.to_csv(output_dir / 'confidence_compliance_results.csv', index=False)
        print(f"\nConfidence compliance results saved to: {output_dir / 'confidence_compliance_results.csv'}")
        
        # Create a summary table for LaTeX
        latex_summary = create_confidence_compliance_latex_table(confidence_df)
        with open(output_dir / 'confidence_compliance_summary.tex', 'w', encoding='utf-8') as f:
            f.write(latex_summary)
        print(f"LaTeX confidence compliance summary saved to: {output_dir / 'confidence_compliance_summary.tex'}")
    
    return confidence_df

def create_confidence_compliance_latex_table(confidence_df):
    """Create a LaTeX table summarizing confidence compliance results."""
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Confidence Output Compliance Analysis (Integer Requirement)}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\hline")
    latex.append("Prompt & \\makecell{Non-Compliance\\\\Rate (\\%)} & \\makecell{Total\\\\Samples} & \\makecell{Float\\\\Errors} & \\makecell{Text\\\\Errors} & \\makecell{Out of\\\\Range} & \\makecell{Other\\\\Errors} \\\\")
    latex.append("\\hline")
    
    for _, row in confidence_df.iterrows():
        prompt_num = row['Prompt']
        non_comp = row['Confidence_Non_Compliance_Rate']
        total_samples = row['Total_Confidence_Samples']
        float_err = row['Float_Errors']
        text_err = row['Text_Errors']
        range_err = row['Out_Of_Range_Errors']
        other_err = row['Other_Errors']
        
        latex.append(f"{prompt_num} & {non_comp:.3f} & {total_samples} & {float_err} & {text_err} & {range_err} & {other_err} \\\\")
    
    latex.append("\\hline")
    
    # Add overall statistics
    overall_non_comp = (confidence_df['Confidence_Non_Compliant'].sum() / 
                       confidence_df['Total_Confidence_Samples'].sum() * 100)
    total_all = confidence_df['Total_Confidence_Samples'].sum()
    total_float = confidence_df['Float_Errors'].sum()
    total_text = confidence_df['Text_Errors'].sum()
    total_range = confidence_df['Out_Of_Range_Errors'].sum()
    total_other = confidence_df['Other_Errors'].sum()
    
    latex.append(f"\\textbf{{Overall}} & \\textbf{{{overall_non_comp:.3f}}} & \\textbf{{{total_all}}} & \\textbf{{{total_float}}} & \\textbf{{{total_text}}} & \\textbf{{{total_range}}} & \\textbf{{{total_other}}} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return '\n'.join(latex)

# Main analysis function for a single model
def analyze_model(df, model_name, prompts, output_dir):
    """Perform complete analysis for a single model"""
    print(f"\nStarting analysis for {model_name}...")
    
    # Check if we have enough data
    if len(df) < 100:
        print(f"WARNING: Only {len(df)} rows available for {model_name}. Statistical tests may be unreliable.")
        print("Skipping detailed statistical analysis for this model due to insufficient data.")
        
        # Still save basic summary
        summary_df = pd.DataFrame([{
            'Model': model_name,
            'Total Rows': len(df),
            'Status': 'Insufficient data for analysis'
        }])
        summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
        return
    
    # Calculate relative probabilities
    df['Total_Prob'] = df['Token_1_Prob'] + df['Token_2_Prob']
    
    # Handle division by zero and create relative probability
    # Use np.where to avoid division by zero warnings
    df['Relative_Prob'] = np.where(
        df['Total_Prob'] > 0,
        df['Token_1_Prob'] / df['Total_Prob'],
        np.nan
    )
    
    # Check for any infinite or NaN values and report them
    non_finite_mask = ~np.isfinite(df['Relative_Prob'])
    if non_finite_mask.any():
        num_non_finite = non_finite_mask.sum()
        print(f"Warning: Found {num_non_finite} non-finite relative probability values for {model_name}")
        print(f"  - NaN values: {df['Relative_Prob'].isna().sum()}")
        print(f"  - Inf values: {np.isinf(df['Relative_Prob']).sum()}")
        
        # Optionally, you can investigate which prompts have issues
        if num_non_finite < 20:  # Only show details for small numbers
            problematic_rows = df[non_finite_mask][['Original Main Part', 'Token_1_Prob', 'Token_2_Prob', 'Total_Prob']]
            print("Problematic rows:")
            print(problematic_rows.head(10))
    
    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()
    
    # Collect all LaTeX tables
    all_tables = []
    
    # Create summary statistics
    summary_stats = []
    
    # Conduct normality tests and truncated normal tests for all prompts
    normality_results = []
    truncated_results = []
    
    # Process each unique original prompt
    for idx, original_prompt in enumerate(unique_prompts):
        prompt_data = df[df['Original Main Part'] == original_prompt]
        prompt_info = prompts[idx]
        token_options = prompt_info[2]
        
        # Create histograms
        create_probability_histogram(prompt_data, idx, token_options, output_dir / "figures")
        create_confidence_histogram(prompt_data, idx, token_options, output_dir / "figures")
        
        # Create LaTeX table
        latex_table = create_latex_table(prompt_data, idx, prompt_info, output_dir)
        all_tables.append(latex_table)
        
        # Calculate statistics
        first_token = token_options[0]
        second_token = token_options[1]
        
        # Filter out non-finite values for statistics calculation
        finite_prob_data = prompt_data[np.isfinite(prompt_data['Relative_Prob'])]
        
        # Calculate percentiles for 95% interval
        if len(finite_prob_data) > 0:
            lower_percentile = np.percentile(finite_prob_data['Relative_Prob'], 2.5)
            upper_percentile = np.percentile(finite_prob_data['Relative_Prob'], 97.5)
            mean_prob = finite_prob_data['Relative_Prob'].mean()
            std_prob = finite_prob_data['Relative_Prob'].std()
            min_prob = finite_prob_data['Relative_Prob'].min()
            max_prob = finite_prob_data['Relative_Prob'].max()
        else:
            lower_percentile = np.nan
            upper_percentile = np.nan
            mean_prob = np.nan
            std_prob = np.nan
            min_prob = np.nan
            max_prob = np.nan
        
        # Calculate confidence statistics if available
        has_confidence_data = 'Weighted Confidence' in prompt_data.columns and not prompt_data['Weighted Confidence'].isna().all()
        
        stats = {
            'Prompt Number': idx + 1,
            'First Token': first_token,
            'Second Token': second_token,
            f'Mean Relative Probability of "{first_token}"': mean_prob,
            'Std Dev': std_prob,
            'Min': min_prob,
            'Max': max_prob,
            '2.5th Percentile': lower_percentile,
            '97.5th Percentile': upper_percentile,
            '95% Interval Width': upper_percentile - lower_percentile if not np.isnan(upper_percentile) else np.nan
        }
        
        # Add confidence statistics if available
        if has_confidence_data:
            filtered_data = prompt_data.dropna(subset=['Weighted Confidence'])
            if len(filtered_data) > 0:
                conf_lower_percentile = np.percentile(filtered_data['Weighted Confidence'], 2.5)
                conf_upper_percentile = np.percentile(filtered_data['Weighted Confidence'], 97.5)
                
                stats.update({
                    f'Mean Weighted Confidence for "{first_token}"': filtered_data['Weighted Confidence'].mean(),
                    'Confidence Std Dev': filtered_data['Weighted Confidence'].std(),
                    'Confidence Min': filtered_data['Weighted Confidence'].min(),
                    'Confidence Max': filtered_data['Weighted Confidence'].max(),
                    'Confidence 2.5th Percentile': conf_lower_percentile,
                    'Confidence 97.5th Percentile': conf_upper_percentile,
                    'Confidence 95% Interval Width': conf_upper_percentile - conf_lower_percentile
                })
        
        summary_stats.append(stats)
        
        # Conduct normality tests on relative probability
        prob_results = conduct_normality_tests(prompt_data, 'Relative_Prob', idx)
        prob_results['Column'] = 'Relative_Prob'
        normality_results.append(prob_results)
        
        # Create QQ-plot for relative probability
        create_qq_plot(prompt_data, 'Relative_Prob', idx, token_options, output_dir / "figures")
        
        # Conduct truncated normal test for relative probability
        try:
            trunc_results, simulated_data = conduct_truncated_normal_test(prompt_data, 'Relative_Prob', idx)
            truncated_results.append(trunc_results)
            
            # Create comparison plot for truncated model
            create_truncated_model_plot(prompt_data, 'Relative_Prob', idx, token_options, simulated_data, 
                                      output_dir / "figures", trunc_results['KS Statistic'])
        except Exception as e:
            print(f"Error in truncated normal test for prompt {idx+1} Relative_Prob: {str(e)}")
        
        # Check if we have confidence data and test it too
        if has_confidence_data:
            filtered_data = prompt_data.dropna(subset=['Weighted Confidence'])
            if len(filtered_data) > 0:
                # Conduct normality tests on confidence data
                conf_results = conduct_normality_tests(filtered_data, 'Weighted Confidence', idx)
                conf_results['Column'] = 'Weighted_Confidence'
                normality_results.append(conf_results)
                
                # Create QQ-plot for confidence data
                create_qq_plot(filtered_data, 'Weighted Confidence', idx, token_options, output_dir / "figures")
                
                # Conduct truncated normal test for confidence data
                # First, rescale confidence data to [0, 1] range if needed
                if filtered_data['Weighted Confidence'].max() > 1:
                    # Assuming confidence data is in range [0, 100]
                    rescaled_data = filtered_data.copy()
                    rescaled_data['Weighted Confidence'] = rescaled_data['Weighted Confidence'] / 100.0
                    
                    try:
                        conf_trunc_results, conf_simulated_data = conduct_truncated_normal_test(
                            rescaled_data, 'Weighted Confidence', idx)
                        
                        # Update the simulated data to original scale for plotting
                        conf_simulated_data = conf_simulated_data * 100.0
                        
                        # Adjust the results to reflect the original scale
                        conf_trunc_results['Underlying Normal Mean'] *= 100.0
                        conf_trunc_results['Underlying Normal Std Dev'] *= 100.0
                        conf_trunc_results['Observed Mean'] *= 100.0
                        conf_trunc_results['Observed Std Dev'] *= 100.0
                        conf_trunc_results['Simulated Mean'] *= 100.0
                        conf_trunc_results['Simulated Std Dev'] *= 100.0
                        conf_trunc_results['Interior Mean'] *= 100.0
                        conf_trunc_results['Interior Std Dev'] *= 100.0
                        
                        truncated_results.append(conf_trunc_results)
                        
                        # Create comparison plot for truncated model
                        create_truncated_model_plot(filtered_data, 'Weighted Confidence', idx, 
                                                   token_options, conf_simulated_data, output_dir / "figures", 
                                                   conf_trunc_results['KS Statistic'])
                    except Exception as e:
                        print(f"Error in truncated normal test for prompt {idx+1} Weighted Confidence: {str(e)}")
    
    # Create the LaTeX content with just the tables
    latex_content = '\n'.join(all_tables)
    
    # Save only the tables content without any document structure
    with open(output_dir / 'prompt_perturbation_tables.tex', 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    # Also save a standalone LaTeX document
    standalone_document = create_standalone_latex_document(all_tables)
    with open(output_dir / 'prompt_perturbation_standalone.tex', 'w', encoding='utf-8') as f:
        f.write(standalone_document)
    
    # Create summary statistics table
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    
    # Create the combined visualizations
    create_combined_visualization(df, prompts, output_dir)
    create_combined_confidence_visualization(df, prompts, output_dir)
    
    # Create DataFrame from normality test results and save
    if normality_results:
        normality_df = pd.DataFrame(normality_results)
        normality_df.to_csv(output_dir / 'normality_test_results.csv', index=False)
    
    # Create DataFrame from truncated normal test results and save
    if truncated_results:
        truncated_df = pd.DataFrame(truncated_results)
        truncated_df.to_csv(output_dir / 'truncated_normal_test_results.csv', index=False)
    
    # Calculate Cohen's kappa
    kappa, observed_agreement, expected_agreement = calculate_cohens_kappa(df)
    
    # Save kappa results to a summary file
    kappa_summary = {
        'Model': model_name,
        'Cohen\'s Kappa': kappa,
        'Observed Agreement': observed_agreement,
        'Expected Agreement': expected_agreement
    }
    pd.DataFrame([kappa_summary]).to_csv(output_dir / 'cohens_kappa_results.csv', index=False)
    
    # Check output compliance with instructions
    compliance_df = check_output_compliance(df, prompts, output_dir)
    
    # Check confidence compliance
    confidence_df = check_confidence_compliance(df, prompts, output_dir)
    
    print(f"\nAnalysis complete for {model_name}!")
    print(f"Results saved to: {output_dir}")

# Main script
if __name__ == "__main__":
    # Read the results from the multi-model analysis
    results_file = "G:/My Drive/Computational/llm_interpretation/results_30_multi_model.xlsx"
    print(f"Reading results from: {results_file}")
    df = pd.read_excel(results_file)
    
    # Save combined results to a new Excel file
    combined_results_file = "G:/My Drive/Computational/llm_interpretation/combined_results.xlsx"
    df.to_excel(combined_results_file, index=False)
    
    # Define prompts list (used for all models)
    prompts = [
        ("An insurance policy contains the following exclusion: 'We will not pay for loss or damage caused directly or indirectly by flood, surface water, waves, tides, tidal waves, overflow of any body of water, or their spray, all whether driven by wind or not.' If an insured files a claim for damage caused by a levee failure rather than natural causes, does the exclusion apply? Please analyze whether the insurance policy would cover the damage or not.", "Answer only 'Covered' if insurance covers the loss or 'Not Covered' if it does not, without any other explanation.", ("Covered", "Not")),
        ("In a prenuptial agreement, if one party files a petition for divorce after seven years, withdraws it without serving it, and then files again after ten years, which filing date determines the number of years of marriage for calculating the payment amount? The agreement states that payment will be based on 'the number of full years they have been married at the time a Petition for Dissolution of Marriage is filed.' Please analyze whether 'a Petition' refers to the first or ultimate petition filed.", "Answer only 'First Petition' if the first filing date is used or 'Ultimate Petition' if the ultimate filing date is used, without any other explanation.", ("Ultimate", "First")),
        ("Does the following contract term from 1961 naturally include only existing affiliates at the time of contract, or does it potentially encompass affiliates that might be created over time? The term binds [Company] and its 'other affiliate[s]' to a 50/50 royalty split after deducting fees charged by third parties that intermediate in foreign markets. Please analyze whether the term 'other affiliate[s]' includes only existing affiliates or includes future affiliates as well.", "Answer only 'Existing Affiliates' or 'Future Affiliates', without any other explanation.", ("Existing", "Future")),
        ("A contractor and business corresponded about construction of a new foundry. The contractor offered to do the job either by offering an itemized list or charging cost + 10%. After a phone call where they allegedly agreed payment would be made 'in the usual manner', the foundry accepted in writing. If one party claims it is customary to pay 85% of payments due at the end of every month, but the other argues payments are only due upon substantial completion, how should the term 'usual manner' be interpreted? Does this term refer to the monthly installment payments or to payment upon completion?", "Answer only 'Monthly Installment Payments' or 'Payment Upon Completion', without any other explanation.", ("Monthly", "Payment")),
        ("You are analyzing an insurance policy dispute. The policy states: '[Insurer will pay for] the felonious abstraction of insured property (1) from within the premises by a person making felonious entry therein by actual force and violence, of which force and violence there are visible marks made by tools, explosives, electricity or chemicals.' A business has experienced a theft where there is clear evidence that a third party committed the burglary. No inside job is suspected. Based on these terms, would this policy provide compensation for losses resulting from this substantiated third-party burglary? Please analyze whether coverage would be provided.", "Answer only 'Covered' if insurance covers the loss or 'Not Covered' if it does not, without any other explanation.", ("Covered", "Not"))
    ]
    
    # Check if Model column exists and print available models
    if 'Model' in df.columns:
        print(f"\nAvailable models in the dataset:")
        available_models = df['Model'].unique()
        for model in available_models:
            print(f"  - {model}")
        print(f"\nTotal rows: {len(df)}")
        
        # Analyze each model separately by default
        print("\nAnalyzing each model separately...")
        
        # Process each model
        for model_name in available_models:
            print(f"\n{'='*60}")
            print(f"ANALYZING MODEL: {model_name}")
            print(f"{'='*60}")
            
            # Filter data for current model
            model_df = df[df['Model'] == model_name].copy()
            print(f"Processing {len(model_df)} rows for {model_name}")
            
            # Update output directory to include model name
            model_safe_name = model_name.replace('.', '_').replace('-', '_')
            output_dir = Path(f"G:/My Drive/Computational/llm_interpretation/output/{model_safe_name}")
            output_dir.mkdir(parents=True, exist_ok=True)
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            # Analyze this model
            analyze_model(model_df, model_name, prompts, output_dir)
    else:
        print("\nNo 'Model' column found - analyzing as single model dataset")
        
        # Keep original output directory for single model case
        output_dir = Path("G:/My Drive/Computational/llm_interpretation/output")
        output_dir.mkdir(exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Analyze as single model
        analyze_model(df, "Single Model", prompts, output_dir)
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE!")
    print("="*60) 