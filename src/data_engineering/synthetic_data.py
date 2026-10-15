import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_policies=10000, output_dir="data/raw"):
    """Generate synthetic PIT tables for the insurance rating system."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42) # For reproducibility
    
    # 1. Table: Policy
    policy_ids = [f"POL-{100000 + i}" for i in range(num_policies)]
    df_policy = pd.DataFrame({
        "policy_id": policy_ids,
        "policy_term_months": np.random.choice([6, 12], size=num_policies),
        "state": np.random.choice(["CA", "TX", "NY", "FL", "IL"], size=num_policies)
    })
    
    # 2. Table: Vehicle
    # Assume 1 vehicle per policy for simplicity; this can be expanded
    vehicle_ids = [f"VEH-{100000 + i}" for i in range(num_policies)]
    df_vehicle = pd.DataFrame({
        "vehicle_id": vehicle_ids,
        "policy_id": policy_ids,
        "vehicle_age": np.random.randint(0, 20, size=num_policies),
        "vehicle_symbol": np.random.randint(10, 25, size=num_policies), # Factor de riesgo del auto
        "annual_mileage": np.random.randint(5000, 25000, size=num_policies)
    })
    
    # 3. Table: Driver
    driver_ids = [f"DRV-{100000 + i}" for i in range(num_policies)]
    df_driver = pd.DataFrame({
        "driver_id": driver_ids,
        "policy_id": policy_ids,
        "driver_age": np.random.randint(18, 80, size=num_policies),
        "credit_score": np.random.randint(500, 850, size=num_policies),
        "years_licensed": np.random.randint(1, 60, size=num_policies)
    })
    
    # 4. Table: Incident (Claims / History)
    df_incident = pd.DataFrame({
        "policy_id": policy_ids,
        "num_claims_past_3_years": np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.08, 0.02], size=num_policies),
        "num_moving_violations": np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05], size=num_policies)
    })
    
    # Guardar en CSV simulando las tablas PIT de entrada
    df_policy.to_csv(f"{output_dir}/policy.csv", index=False)
    df_vehicle.to_csv(f"{output_dir}/vehicle.csv", index=False)
    df_driver.to_csv(f"{output_dir}/driver.csv", index=False)
    df_incident.to_csv(f"{output_dir}/incident.csv", index=False)
    
    print(f"✅ {num_policies} synthetic policies generated in '{output_dir}'")

if __name__ == "__main__":
    generate_synthetic_data()