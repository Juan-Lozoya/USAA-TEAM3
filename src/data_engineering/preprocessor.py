import pandas as pd
import os

class DataPreprocessor:
    def __init__(self, input_dir=None):
        # Por defecto, usar la carpeta `data/raw` relativa a este archivo
        if input_dir:
            self.input_dir = input_dir
        else:
            base_dir = os.path.dirname(__file__)
            self.input_dir = os.path.join(base_dir, "data", "raw")
        
    def load_and_merge_data(self):
        """Read the separate PIT tables and merge them into an Analytical Base Table (ABT)."""
        print("Starting data engineering pipeline...")
        
        try:
            # Verify input directory exists
            print(f"Searching for files in: {self.input_dir}")
            if not os.path.isdir(self.input_dir):
                raise FileNotFoundError(f"Directory does not exist: {self.input_dir}")

            # Load tables
            df_policy = pd.read_csv(os.path.join(self.input_dir, "policy.csv"))
            df_vehicle = pd.read_csv(os.path.join(self.input_dir, "vehicle.csv"))
            df_driver = pd.read_csv(os.path.join(self.input_dir, "driver.csv"))
            df_incident = pd.read_csv(os.path.join(self.input_dir, "incident.csv"))
            
            # Simple ID validation
            assert df_policy['policy_id'].is_unique, "Duplicate policy_ids found in Policy table"
            
            # Joins (Merge)
            # Empezamos con Policy y unimos Vehicle (1:1 o 1:N)
            abt = df_policy.merge(df_vehicle, on="policy_id", how="left")
            
            # Unimos Driver
            abt = abt.merge(df_driver, on="policy_id", how="left")
            
            # Unimos Incidents
            abt = abt.merge(df_incident, on="policy_id", how="left")
            
            # Handle missing data (basic imputation)
            abt['num_claims_past_3_years'] = abt['num_claims_past_3_years'].fillna(0)
            abt['num_moving_violations'] = abt['num_moving_violations'].fillna(0)
            print(f"✅ Pipeline completed. Final table shape: {abt.shape}")
            return abt
            
        except FileNotFoundError as e:
            # Provide additional information to help debug paths
            current_dir = os.path.dirname(__file__)
            available = []
            try:
                available = os.listdir(current_dir)
            except Exception:
                available = [f"Could not list {current_dir}"]

            print(f"❌ Error: Source files not found. {e}")
            print(f"Searching relative to: {current_dir}")
            print(f"Contents of {current_dir}: {available}")
            return None

if __name__ == "__main__":
    # Quick test of the preprocessor
    preprocessor = DataPreprocessor()
    df_final = preprocessor.load_and_merge_data()
    if df_final is not None:
        print(df_final.head())