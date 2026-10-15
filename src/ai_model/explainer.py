import pandas as pd
import xgboost as xgb
import shap
from sklearn.preprocessing import LabelEncoder
import numpy as np

class PremiumAIExplainer:
    def __init__(self):
        # Configure an XGBoost regression model
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.label_encoders = {}
        
        # Input variables for the model (features)
        self.features = [
            'policy_term_months', 'vehicle_age', 'vehicle_symbol', 
            'annual_mileage', 'driver_age', 'credit_score', 
            'years_licensed', 'num_claims_past_3_years', 
            'num_moving_violations', 'state'
        ]
        self.target = 'final_premium'

    def preprocess_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features, encoding categorical variables."""
        X = df[self.features].copy()
        y = df[self.target].copy()

        # Encode the 'state' column to numeric values
        le = LabelEncoder()
        X['state'] = le.fit_transform(X['state'])
        self.label_encoders['state'] = le

        return X, y

    def train_and_explain(self, rated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Train the XGBoost model and compute SHAP values.
        Returns the original DataFrame concatenated with SHAP contributions.
        """
        print("Preparing data for AI model...")
        X, y = self.preprocess_features(rated_df)

        print("Training XGBoost model...")
        self.model.fit(X, y)

        # Evaluate basic accuracy (R^2)
        score = self.model.score(X, y)
        print(f"Model accuracy (R^2): {score:.4f}")

        print("Calculating explainability values (SHAP)...")
        # TreeExplainer is highly optimized for tree-based models like XGBoost
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value

        # If expected_value is an array (depending on SHAP version), take the scalar
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]

        # Create a DataFrame with SHAP values
        # Each column will be named 'shap_<feature_name>'
        shap_cols = [f"shap_{col}" for col in self.features]
        df_shap = pd.DataFrame(shap_values, columns=shap_cols, index=rated_df.index)
        
        # Add base value (global predictive mean)
        df_shap['shap_base_value'] = expected_value
        
        # Validar la propiedad aditiva de SHAP: base_value + sum(shap_values) = prediction
        predictions = self.model.predict(X)
        df_shap['ai_predicted_premium'] = predictions.round(2)

        # Concatenate the original DataFrame with the SHAP explanations
        final_explained_df = pd.concat([rated_df, df_shap], axis=1)
        
        print("✅ SHAP explainability generated successfully.")
        return final_explained_df

if __name__ == "__main__":
    import sys
    import os
    
    # Local imports of previous modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data_engineering.preprocessor import DataPreprocessor
    from src.rating_engine.calculator import RatingCalculator
    
    # 1. Data pipeline
    preprocessor = DataPreprocessor()
    df_abt = preprocessor.load_and_merge_data()
    
    # 2. Rating Engine
    calculator = RatingCalculator()
    df_rated = calculator.calculate_premiums(df_abt)
    
    # 3. AI & SHAP
    if df_rated is not None:
        ai_explainer = PremiumAIExplainer()
        df_explained = ai_explainer.train_and_explain(df_rated)
        
        # Show SHAP breakdown for the first policy
        print("\nSHAP impact example for policy 0:")
        policy_0 = df_explained.iloc[0]
        print(f"Actual Premium (Rating Engine): ${policy_0['final_premium']}")
        print(f"Predicted Premium (AI): ${policy_0['ai_predicted_premium']}")
        print(f"Base Value (Mean): ${policy_0['shap_base_value']:.2f}")
        
        print("\nImpact per feature (SHAP):")
        for feature in ai_explainer.features:
            print(f" - {feature}: ${policy_0[f'shap_{feature}']:.2f}")