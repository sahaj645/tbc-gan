import numpy as np
from scipy.stats import norm

# Defining constants and parameters
NUM_CLOCKS = 3  # Number of clocks in the ensemble (e.g., MASER A, MASER B, EFOS MASER)
LEARNING_RATE = 0.01  # Learning rate 'eta' for perceptron weight updates, between 0 and 1
WINDOW_SIZE = 2.5 * 24 * 3600  # 2.5 days window for weight dataset generation, in seconds
SIMULATION_DAYS = 30  # Simulation duration in days
SECONDS_PER_DAY = 24 * 3600
TOTAL_SECONDS = SIMULATION_DAYS * SECONDS_PER_DAY
AVERAGING_TIMES = [1000, 100000]  # Averaging times for weight calculation (in seconds)

# Simulating phase difference data (placeholder for real IRNWT data)
def generate_phase_differences(num_clocks, total_seconds):
    time = np.arange(total_seconds)
    phase_diffs = []
    for i in range(num_clocks):
        # Simulating phase differences with noise (placeholder for real data)
        noise = norm.rvs(scale=1e-9, size=total_seconds)  # Nanosecond-scale noise
        phase_diff = np.cumsum(noise)  # Cumulative phase difference
        phase_diffs.append(phase_diff)
    return np.array(phase_diffs)

# Calculating Overlapped Allan Variance for a single clock
def overlapped_allan_variance(phase_data, tau, rate=1):
    n = len(phase_data)
    tau_rate = int(tau * rate)
    max_m = int(np.floor((n - 2) / tau_rate))  # Adjusted to prevent index out of bounds
    if max_m <= 0:  # Checking if valid number of samples exist
        return 0
    sum_squares = 0
    for i in range(max_m):
        idx1 = i * tau_rate
        idx2 = (i + 1) * tau_rate
        idx3 = (i + 2) * tau_rate
        if idx3 >= n:  # Ensuring index does not exceed array bounds
            break
        x1 = phase_data[idx1]
        x2 = phase_data[idx2]
        x3 = phase_data[idx3]
        diff = (x3 - 2 * x2 + x1) ** 2
        sum_squares += diff
    return sum_squares / (2 * max_m) if max_m > 0 else 0

# Calculating weights using Inverse Allan Variance
def calculate_weights(phase_diffs, averaging_time):
    variances = []
    for i in range(phase_diffs.shape[0]):
        var = overlapped_allan_variance(phase_diffs[i], averaging_time)
        variances.append(var if var > 0 else 1e-20)
    inv_squares = [1 / (v if v > 0 else 1e-20) for v in variances]
    total_inv_squares = sum(inv_squares)
    weights = [inv / total_inv_squares for inv in inv_squares]
    return np.array(weights)

# Computing ensemble timescale (S - C)
def compute_ensemble_timescale(phase_diffs, weights, ref_clock=0):
    ensemble = np.zeros_like(phase_diffs[0])
    for t in range(len(ensemble)):
        sum_weighted_diff = 0
        for i in range(len(phase_diffs)):
            diff = phase_diffs[i][t] - phase_diffs[ref_clock][t]  # C_i - C
            sum_weighted_diff += weights[i] * diff
        ensemble[t] = sum_weighted_diff  # S - C
    return ensemble

# Perceptron learning rule for weight updates
def update_weights(weights, actual_output, desired_output, phase_diffs, ref_clock=0):
    error = desired_output - actual_output
    new_weights = weights.copy()
    if not np.isclose(actual_output, desired_output, rtol=1e-5):
        for i in range(len(weights)):
            new_weights[i] = weights[i] - LEARNING_RATE * error
    # Normalizing weights to sum to 1
    total = sum(new_weights)
    if total > 0:
        new_weights = new_weights / total
    return new_weights

# Main ANN-based ensemble algorithm
def ann_ensemble_algorithm():
    # Generating or loading phase difference data
    phase_diffs = generate_phase_differences(NUM_CLOCKS, TOTAL_SECONDS)
    
    # Initializing weights and results storage
    initial_weights = {}
    ensemble_outputs = {}
    
    # Processing for each averaging time
    for tau in AVERAGING_TIMES:
        # Initializing weights using inverse Allan variance
        weights = calculate_weights(phase_diffs, tau)
        initial_weights[tau] = weights.copy()
        
        # Computing initial ensemble timescale
        ensemble = compute_ensemble_timescale(phase_diffs, weights)
        
        # Simulating desired output (placeholder: assuming better stability target)
        desired_output = 1e-16  # Target stability for 1 day, as a placeholder
        
        # Iterating over a window for learning
        for t in range(0, TOTAL_SECONDS, int(WINDOW_SIZE)):
            window_end = min(t + int(WINDOW_SIZE), TOTAL_SECONDS)
            window_phase = phase_diffs[:, t:window_end]
            
            # Computing actual output (stability of ensemble)
            actual_output = overlapped_allan_variance(ensemble[t:window_end], 24 * 3600)
            
            # Updating weights using perceptron learning rule
            weights = update_weights(weights, actual_output, desired_output, window_phase)
            
            # Recomputing ensemble with updated weights
            ensemble[t:window_end] = compute_ensemble_timescale(window_phase, weights)
        
        # Storing final ensemble output
        ensemble_outputs[tau] = ensemble
    
    # Returning results
    return initial_weights, ensemble_outputs

# Running the algorithm
if __name__ == "__main__":
    initial_weights, ensemble_outputs = ann_ensemble_algorithm()
    
    
    for tau in AVERAGING_TIMES:
        print(f"Averaging Time: {tau} seconds")
        print(f"Initial Weights: {initial_weights[tau]}")
        print(f"Final Ensemble Stability (1 sec): {overlapped_allan_variance(ensemble_outputs[tau], 1)}")
        print(f"Final Ensemble Stability (1 day): {overlapped_allan_variance(ensemble_outputs[tau], 24 * 3600)}")