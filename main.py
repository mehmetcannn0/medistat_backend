from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
from services.stats_service import StatsService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_df(file: UploadFile):
    contents = await file.read()
    if file.filename.endswith('.csv'):
        # Virgül veya noktalı virgül ayrımını otomatik yapar
        return pd.read_csv(io.BytesIO(contents), sep=None, engine='python')
    else:
        return pd.read_excel(io.BytesIO(contents))

@app.post("/upload-summary")
async def get_summary(file: UploadFile = File(...)):
    try:
        df = await load_df(file)
        return {
            "status": "success",
            "data": {
                "row_count": len(df),
                "columns": df.columns.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-stats")
async def calculate_stats(file: UploadFile = File(...), columns: str = Form(""), methods: str = Form("")):
    try:
        # Flutter'dan gelen virgüllü stringi listeye çeviriyoruz
        selected_cols = [c.strip() for c in columns.split(",") if c.strip()]
        selected_methods = [m.strip() for m in methods.split(",") if m.strip()]
        
        df = await load_df(file)
        final_table_results = []

        # 1. Tanımlayıcı İstatistikler
        if "descriptive" in selected_methods:
            stats_data = StatsService.calculate_descriptive(df, selected_cols)
            for col, val in stats_data.items():
                final_table_results.append({
                    "Analiz": "Tanımlayıcı",
                    "Parametre": col,
                    "Değer 1": f"Ort: {val['Ortalama']}",
                    "Değer 2": f"Std: {val['Std. Sapma']}",
                    "Yorum": f"Medyan: {val['Medyan']} (n={val['Adet']})"
                })

        # 2. T-Test
        if "ttest" in selected_methods:
            ttest_data = StatsService.calculate_t_test(df, selected_cols)
            for pair, val in ttest_data.items():
                final_table_results.append({
                    "Analiz": "T-Test",
                    "Parametre": pair,
                    "Değer 1": f"p: {val['p_değeri']}",
                    "Değer 2": val['Sonuç'],
                    "Yorum": "Gruplar arası fark"
                })

        # 3. Korelasyon
        if "correlation" in selected_methods:
            corr_data = StatsService.calculate_correlation(df, selected_cols)
            for col1, relations in corr_data.items():
                for col2, value in relations.items():
                    if col1 < col2:
                        final_table_results.append({
                            "Analiz": "Korelasyon",
                            "Parametre": f"{col1} & {col2}",
                            "Değer 1": f"r: {round(value, 3)}",
                            "Değer 2": "Güçlü" if abs(value) > 0.7 else "Zayıf/Orta",
                            "Yorum": "İlişki Katsayısı"
                        })

        # 4. Lojistik Regresyon
        if "logistic" in selected_methods:
            logistic_data = StatsService.calculate_logistic_regression(df, selected_cols)
            for var, val in logistic_data.items():
                final_table_results.append({
                    "Analiz": "Lojistik Regresyon",
                    "Parametre": f"{var} (Hedef: {selected_cols[-1]})",
                    "Değer 1": f"B: {val['katsayı']}",
                    "Değer 2": f"OR: {val['odds_ratio']}",
                    "Yorum": f"p: {val['p_value']} ({'Anlamlı' if val['p_value'] < 0.05 else 'Anlamsız'})"
                })

        # 5. Chi-Square (Ki-Kare) - YENİ
        if "chisquare" in selected_methods:
            chi_data = StatsService.calculate_chi_square(df, selected_cols)
            for pair, val in chi_data.items():
                final_table_results.append({
                    "Analiz": "Chi-Square",
                    "Parametre": pair,
                    "Değer 1": f"Chi2: {val['Chi2']}",
                    "Değer 2": val['Sonuç'],
                    "Yorum": f"p: {val['p_değeri']}"
                })

        # 6. ANOVA - YENİ
        if "anova" in selected_methods:
            anova_data = StatsService.calculate_anova(df, selected_cols)
            for col, val in anova_data.items():
                final_table_results.append({
                    "Analiz": "ANOVA",
                    "Parametre": col,
                    "Değer 1": f"F: {val['F-Stat']}",
                    "Değer 2": val['Sonuç'],
                    "Yorum": f"p: {val['p_değeri']}"
                })

        # 7. Mann-Whitney U - YENİ
        if "mannwhitney" in selected_methods:
            mwu_data = StatsService.calculate_mann_whitney(df, selected_cols)
            for pair, val in mwu_data.items():
                final_table_results.append({
                    "Analiz": "Mann-Whitney U",
                    "Parametre": pair,
                    "Değer 1": f"U: {val['U-Stat']}",
                    "Değer 2": val['Sonuç'],
                    "Yorum": f"p: {val['p_değeri']}"
                })

        return {"status": "success", "results": final_table_results}

    except Exception as e:
        print(f"Hata detayı: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)