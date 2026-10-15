import pandas as pd
import numpy as np

class RatingCalculator:
    def __init__(self, base_rate=500.0):
        # Standard annual base rate
        self.base_rate = base_rate
        
        # Tablas de factores simples (Lookup tables)
        self.state_factors = {'CA': 1.25, 'NY': 1.30, 'FL': 1.40, 'TX': 1.10, 'IL': 1.05}

    def calculate_premiums(self, abt_df: pd.DataFrame) -> pd.DataFrame:
        """Apply the rating formula and store intermediate calculation steps."""
        df = abt_df.copy()
        print("Calculating exact premiums (Rating Engine)...")

        # 1. Base Rate
        df['calc_base_rate'] = self.base_rate

        # 2. State factor (Geography)
        # If the state is not in the dictionary, use 1.0 by default
        df['calc_state_factor'] = df['state'].map(self.state_factors).fillna(1.0)

        # 3. Driver Age Factor
        # Under 25 pay more (1.6x), over 65 (1.2x), others (1.0x)
        conditions_age = [
            (df['driver_age'] < 25),
            (df['driver_age'] > 65)
        ]
        choices_age = [1.6, 1.2]
        df['calc_age_factor'] = np.select(conditions_age, choices_age, default=1.0)

        # 4. Vehicle Symbol Factor (vehicle risk)
        # Normalized symbol (e.g., symbol 15 -> factor 1.15)
        df['calc_vehicle_factor'] = 1.0 + ((df['vehicle_symbol'] - 10) / 100)

        # 5. Claims History Factor
        # +25% surcharge per claim in the last 3 years
        df['calc_claims_surcharge'] = 1.0 + (df['num_claims_past_3_years'] * 0.25)

        # 6. Policy Term Factor (6 vs 12 months)
        # A 6-month policy costs half of the annualized amount
        df['calc_term_factor'] = df['policy_term_months'] / 12.0

        # ==========================================
        # FINAL PREMIUM CALCULATION (Multiplicative)
        # ==========================================
        df['final_premium'] = (
            df['calc_base_rate'] *
            df['calc_state_factor'] *
            df['calc_age_factor'] *
            df['calc_vehicle_factor'] *
            df['calc_claims_surcharge'] *
            df['calc_term_factor']
        )
        
        # Round to 2 decimals for currency
        df['final_premium'] = df['final_premium'].round(2)

        print(f"✅ Calculation completed. Average premium: ${df['final_premium'].mean():.2f}")
        return df

if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path to import the preprocessor
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data_engineering.preprocessor import DataPreprocessor
    
    # 1. Load data
    preprocessor = DataPreprocessor()
    df_abt = preprocessor.load_and_merge_data()
    
    # 2. Calculate premiums
    if df_abt is not None:
        calculator = RatingCalculator()
        df_rated = calculator.calculate_premiums(df_abt)
        
        # Show an example breakdown for the first policy
        print("\nExample calculation breakdown for policy 0:")
        cols_to_show = ['policy_id', 'final_premium'] + [c for c in df_rated.columns if c.startswith('calc_')]
        print(df_rated[cols_to_show].head(1).T)