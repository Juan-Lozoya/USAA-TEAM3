import pandas as pd
import os

class DataPreprocessor:
    def __init__(self, input_dir="data/raw"):
        self.input_dir = input_dir
        
    def load_and_merge_data(self):
        """Lee las tablas PIT separadas y las une en una Analytical Base Table (ABT)."""
        print("Iniciando Data Engineering pipeline...")
        
        try:
            # Cargar tablas
            df_policy = pd.read_csv(os.path.join(self.input_dir, "policy.csv"))
            df_vehicle = pd.read_csv(os.path.join(self.input_dir, "vehicle.csv"))
            df_driver = pd.read_csv(os.path.join(self.input_dir, "driver.csv"))
            df_incident = pd.read_csv(os.path.join(self.input_dir, "incident.csv"))
            
            # Validación simple de IDs
            assert df_policy['policy_id'].is_unique, "Existen policy_ids duplicados en la tabla Policy"
            
            # Joins (Merge)
            # Empezamos con Policy y unimos Vehicle (1:1 o 1:N)
            abt = df_policy.merge(df_vehicle, on="policy_id", how="left")
            
            # Unimos Driver
            abt = abt.merge(df_driver, on="policy_id", how="left")
            
            # Unimos Incidents
            abt = abt.merge(df_incident, on="policy_id", how="left")
            
            # Manejo de datos nulos (Imputación básica)
            abt['num_claims_past_3_years'] = abt['num_claims_past_3_years'].fillna(0)
            abt['num_moving_violations'] = abt['num_moving_violations'].fillna(0)
            
            print(f"✅ Pipeline completado. Shape de la tabla final: {abt.shape}")
            return abt
            
        except FileNotFoundError as e:
            print(f"❌ Error: No se encontraron los archivos fuente. {e}")
            return None

if __name__ == "__main__":
    # Prueba rápida del preprocesador
    preprocessor = DataPreprocessor()
    df_final = preprocessor.load_and_merge_data()
    if df_final is not None:
        print(df_final.head())