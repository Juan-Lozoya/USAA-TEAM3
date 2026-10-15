import pandas as pd
import os

class ExcelReportGenerator:
    def __init__(self, output_dir="data/output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_workbook(self, df: pd.DataFrame, filename="explainability_report.xlsx"):
        filepath = os.path.join(self.output_dir, filename)
        print(f"Generando Excel interactivo en {filepath}...")

        # Iniciar el motor XlsxWriter
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        
        # 1. Volcar todas las pólizas en la primera pestaña
        sheet_data_name = 'All_Policies'
        df.to_excel(writer, sheet_name=sheet_data_name, index=False)
        
        workbook = writer.book
        worksheet_dash = workbook.add_worksheet('Explainability_Dashboard')
        
        # Formatos visuales básicos
        title_format = workbook.add_format({'bold': True, 'font_size': 14, 'bg_color': '#4F81BD', 'font_color': 'white'})
        header_format = workbook.add_format({'bold': True, 'bg_color': '#DCE6F1', 'border': 1})
        money_format = workbook.add_format({'num_format': '$#,##0.00'})

        # =========================================================
        # CONSTRUCCIÓN DEL DASHBOARD INTERACTIVO
        # =========================================================
        worksheet_dash.set_column('A:A', 25)
        worksheet_dash.set_column('B:B', 20)

        # Título y Selector de Póliza
        worksheet_dash.write('A1', 'Selector Interactivo de Póliza', title_format)
        worksheet_dash.write('A2', 'ID de Póliza:', header_format)
        
        # Lista desplegable (Data Validation) leyendo desde la pestaña All_Policies
        max_row = len(df) + 1
        worksheet_dash.data_validation('B2', {
            'validate': 'list',
            'source': f"='{sheet_data_name}'!$A$2:$A${max_row}"
        })
        # Valor por defecto inicial para que las fórmulas no muestren error
        worksheet_dash.write('B2', df['policy_id'].iloc[0]) 

        # Generar fórmulas VLOOKUP dinámicamente para cada columna del DataFrame
        # La estructura de VLOOKUP: =VLOOKUP($B$2, All_Policies!$A$2:$ZZ$10005, INDICE_COLUMNA, FALSE)
        current_row = 4
        
        def write_section(title, columns_to_include):
            nonlocal current_row
            worksheet_dash.write(current_row, 0, title, title_format)
            current_row += 1
            
            for col_name in columns_to_include:
                if col_name in df.columns:
                    col_idx = df.columns.get_loc(col_name) + 1 # Excel usa índices base 1
                    
                    worksheet_dash.write(current_row, 0, col_name)
                    # Fórmula VLOOKUP
                    formula = f'=VLOOKUP($B$2, \'{sheet_data_name}\'!$A$2:$ZZ${max_row}, {col_idx}, FALSE)'
                    
                    # Aplicar formato de moneda si es una prima o valor SHAP
                    if 'premium' in col_name or 'shap_' in col_name:
                        worksheet_dash.write_formula(current_row, 1, formula, money_format)
                    else:
                        worksheet_dash.write_formula(current_row, 1, formula)
                    current_row += 1
            current_row += 1 # Espacio entre secciones

        # Agrupar las columnas por sección para mayor claridad en el dashboard
        input_cols = ['state', 'driver_age', 'vehicle_symbol', 'num_claims_past_3_years', 'policy_term_months']
        rating_cols = [c for c in df.columns if c.startswith('calc_')] + ['final_premium']
        shap_cols = ['ai_predicted_premium', 'shap_base_value'] + [c for c in df.columns if c.startswith('shap_')]

        # Escribir las secciones en el dashboard
        write_section("1. Características de Entrada (Inputs)", input_cols)
        write_section("2. Desglose Matemático (Rating Engine)", rating_cols)
        write_section("3. Explicabilidad del Modelo (AI - SHAP)", shap_cols)

        # Cerrar y guardar el archivo
        writer.close()
        print("✅ Workbook generado exitosamente.")

if __name__ == "__main__":
    import sys
    import os
    
    # Simular el flujo completo para probar el Excel
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data_engineering.preprocessor import DataPreprocessor
    from src.rating_engine.calculator import RatingCalculator
    from src.ai_model.explainer import PremiumAIExplainer
    
    df_abt = DataPreprocessor().load_and_merge_data()
    df_rated = RatingCalculator().calculate_premiums(df_abt)
    df_explained = PremiumAIExplainer().train_and_explain(df_rated)
    
    ExcelReportGenerator().generate_workbook(df_explained)