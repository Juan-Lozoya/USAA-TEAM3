import pandas as pd
import xgboost as xgb
import shap
from sklearn.preprocessing import LabelEncoder
import numpy as np

class PremiumAIExplainer:
    def __init__(self):
        # Configuramos un modelo XGBoost de regresión
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.label_encoders = {}
        
        # Variables de entrada para el modelo (Features)
        self.features = [
            'policy_term_months', 'vehicle_age', 'vehicle_symbol', 
            'annual_mileage', 'driver_age', 'credit_score', 
            'years_licensed', 'num_claims_past_3_years', 
            'num_moving_violations', 'state'
        ]
        self.target = 'final_premium'

    def preprocess_features(self, df: pd.DataFrame) -> tuple:
        """Prepara las características, codificando variables categóricas."""
        X = df[self.features].copy()
        y = df[self.target].copy()

        # Codificar la columna 'state' a valores numéricos
        le = LabelEncoder()
        X['state'] = le.fit_transform(X['state'])
        self.label_encoders['state'] = le

        return X, y

    def train_and_explain(self, rated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Entrena el modelo XGBoost y calcula los valores SHAP.
        Devuelve el DataFrame original concatenado con las contribuciones SHAP.
        """
        print("Preparando datos para el modelo de IA...")
        X, y = self.preprocess_features(rated_df)

        print("Entrenando modelo XGBoost...")
        self.model.fit(X, y)

        # Evaluar precisión básica (R^2)
        score = self.model.score(X, y)
        print(f"Precisión del modelo (R^2): {score:.4f}")

        print("Calculando valores de explicabilidad (SHAP)...")
        # TreeExplainer es altamente optimizado para modelos basados en árboles como XGBoost
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value

        # Si expected_value es un array (dependiendo de la versión de SHAP), tomamos el escalar
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]

        # Crear un DataFrame con los valores SHAP
        # Cada columna se llamará 'shap_<nombre_variable>'
        shap_cols = [f"shap_{col}" for col in self.features]
        df_shap = pd.DataFrame(shap_values, columns=shap_cols, index=rated_df.index)
        
        # Agregar el valor base (el promedio predictivo global)
        df_shap['shap_base_value'] = expected_value
        
        # Validar la propiedad aditiva de SHAP: base_value + sum(shap_values) = prediction
        predictions = self.model.predict(X)
        df_shap['ai_predicted_premium'] = predictions.round(2)

        # Unir el DataFrame original con las explicaciones SHAP
        final_explained_df = pd.concat([rated_df, df_shap], axis=1)
        
        print("✅ Explicabilidad SHAP generada con éxito.")
        return final_explained_df

if __name__ == "__main__":
    import sys
    import os
    
    # Imports locales de módulos previos
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data_engineering.preprocessor import DataPreprocessor
    from src.rating_engine.calculator import RatingCalculator
    
    # 1. Pipeline de Datos
    preprocessor = DataPreprocessor()
    df_abt = preprocessor.load_and_merge_data()
    
    # 2. Rating Engine
    calculator = RatingCalculator()
    df_rated = calculator.calculate_premiums(df_abt)
    
    # 3. AI & SHAP
    if df_rated is not None:
        ai_explainer = PremiumAIExplainer()
        df_explained = ai_explainer.train_and_explain(df_rated)
        
        # Mostrar el desglose SHAP para la primera póliza
        print("\nEjemplo de impacto SHAP para la póliza 0:")
        policy_0 = df_explained.iloc[0]
        print(f"Prima Real (Rating Engine): ${policy_0['final_premium']}")
        print(f"Prima Predicha (AI): ${policy_0['ai_predicted_premium']}")
        print(f"Valor Base (Promedio): ${policy_0['shap_base_value']:.2f}")
        
        print("\nImpacto por variable (SHAP):")
        for feature in ai_explainer.features:
            print(f" - {feature}: ${policy_0[f'shap_{feature}']:.2f}")