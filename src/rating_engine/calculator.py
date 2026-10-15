import pandas as pd
import numpy as np

class RatingCalculator:
    def __init__(self, base_rate=500.0):
        # Tarifa base anual estándar
        self.base_rate = base_rate
        
        # Tablas de factores simples (Lookup tables)
        self.state_factors = {'CA': 1.25, 'NY': 1.30, 'FL': 1.40, 'TX': 1.10, 'IL': 1.05}

    def calculate_premiums(self, abt_df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la fórmula de tarificación y guarda los pasos intermedios."""
        df = abt_df.copy()
        print("Calculando primas exactas (Rating Engine)...")

        # 1. Tarifa Base
        df['calc_base_rate'] = self.base_rate

        # 2. Factor por Estado (Geografía)
        # Si el estado no está en el diccionario, usa 1.0 por defecto
        df['calc_state_factor'] = df['state'].map(self.state_factors).fillna(1.0)

        # 3. Factor por Edad del Conductor
        # Menores de 25 pagan más (1.6x), mayores de 65 (1.2x), resto (1.0x)
        conditions_age = [
            (df['driver_age'] < 25),
            (df['driver_age'] > 65)
        ]
        choices_age = [1.6, 1.2]
        df['calc_age_factor'] = np.select(conditions_age, choices_age, default=1.0)

        # 4. Factor por Símbolo del Vehículo (Riesgo del auto)
        # Símbolo normalizado (ej. símbolo 15 -> factor 1.15)
        df['calc_vehicle_factor'] = 1.0 + ((df['vehicle_symbol'] - 10) / 100)

        # 5. Factor por Historial de Siniestros (Claims)
        # +25% de recargo por cada siniestro en los últimos 3 años
        df['calc_claims_surcharge'] = 1.0 + (df['num_claims_past_3_years'] * 0.25)

        # 6. Factor de Término de Póliza (6 vs 12 meses)
        # Una póliza de 6 meses cuesta la mitad de la anualizada
        df['calc_term_factor'] = df['policy_term_months'] / 12.0

        # ==========================================
        # CÁLCULO FINAL DE LA PRIMA (Multiplicativo)
        # ==========================================
        df['final_premium'] = (
            df['calc_base_rate'] *
            df['calc_state_factor'] *
            df['calc_age_factor'] *
            df['calc_vehicle_factor'] *
            df['calc_claims_surcharge'] *
            df['calc_term_factor']
        )
        
        # Redondear a 2 decimales para moneda
        df['final_premium'] = df['final_premium'].round(2)

        print(f"✅ Cálculo completado. Prima promedio: ${df['final_premium'].mean():.2f}")
        return df

if __name__ == "__main__":
    import sys
    import os
    
    # Agregar el root del proyecto al path para importar el preprocesador
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data_engineering.preprocessor import DataPreprocessor
    
    # 1. Cargar datos
    preprocessor = DataPreprocessor()
    df_abt = preprocessor.load_and_merge_data()
    
    # 2. Calcular primas
    if df_abt is not None:
        calculator = RatingCalculator()
        df_rated = calculator.calculate_premiums(df_abt)
        
        # Mostrar un ejemplo del desglose de la primera póliza
        print("\nEjemplo de desglose de cálculo para la póliza 0:")
        cols_to_show = ['policy_id', 'final_premium'] + [c for c in df_rated.columns if c.startswith('calc_')]
        print(df_rated[cols_to_show].head(1).T)