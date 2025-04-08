import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_sample_data(num_samples=1000, failure_rate=0.2):
    """
    Generate sample vehicle maintenance data for testing.
    
    Args:
        num_samples: Number of samples to generate
        failure_rate: Proportion of samples that should indicate maintenance required
    
    Returns:
        DataFrame with synthetic data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='H')
    
    # Generate normal operating parameters
    engine_rpm = np.random.normal(1800, 200, num_samples)
    lub_oil_pressure = np.random.normal(2.5, 0.3, num_samples)
    fuel_pressure = np.random.normal(3.0, 0.4, num_samples)
    coolant_pressure = np.random.normal(1.2, 0.2, num_samples)
    lub_oil_temp = np.random.normal(85, 10, num_samples)
    coolant_temp = np.random.normal(80, 8, num_samples)
    
    # Determine which samples will be failures
    failure_indices = np.random.choice(
        num_samples, 
        size=int(num_samples * failure_rate), 
        replace=False
    )
    
    # Create engine condition array (0 = normal, 1 = maintenance required)
    engine_condition = np.zeros(num_samples)
    engine_condition[failure_indices] = 1
    
    # Modify parameters for failure cases to create correlations
    for idx in failure_indices:
        # Higher engine RPM during failures
        engine_rpm[idx] += np.random.normal(300, 50)
        
        # Lower oil pressure during failures
        lub_oil_pressure[idx] -= np.random.normal(0.8, 0.2)
        
        # Higher temperatures during failures
        lub_oil_temp[idx] += np.random.normal(25, 5)
        coolant_temp[idx] += np.random.normal(20, 5)
        
        # Fluctuating fuel pressure during failures
        fuel_pressure[idx] += np.random.normal(0, 1.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp,
        'Engine Condition': engine_condition.astype(int)
    })
    
    return df

def main():
    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate sample data
    print("Generating sample vehicle maintenance data...")
    df = generate_sample_data(num_samples=5000, failure_rate=0.2)
    
    # Save to CSV
    output_path = data_dir / "vehicle_maintenance_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    print(f"Generated {len(df)} samples with {df['Engine Condition'].sum()} failure cases")

if __name__ == "__main__":
    main()
