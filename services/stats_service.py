import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

class StatsService:
    @staticmethod
    def calculate_descriptive(df: pd.DataFrame, columns: list):
        results = {}
        for col in columns:
            # Veriyi sayıya çevir, hatalıları NaN yap ve temizle
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if not series.empty:
                results[col] = {
                    "Ortalama": round(float(series.mean()), 2),
                    "Medyan": round(float(series.median()), 2),
                    "Std. Sapma": round(float(series.std()), 2),
                    "Adet": int(series.count())
                }
        return results

    @staticmethod
    def calculate_t_test(df: pd.DataFrame, columns: list):
        results = {}
        if len(columns) >= 2:
            col1, col2 = columns[0], columns[1]
            s1 = pd.to_numeric(df[col1], errors='coerce').dropna()
            s2 = pd.to_numeric(df[col2], errors='coerce').dropna()
            
            if not s1.empty and not s2.empty:
                t_stat, p_val = stats.ttest_ind(s1, s2)
                results[f"{col1} vs {col2}"] = {
                    "p_değeri": round(float(p_val), 4),
                    "Sonuç": "Anlamlı" if p_val < 0.05 else "Anlamlı Değil"
                }
        return results

    @staticmethod
    def calculate_correlation(df: pd.DataFrame, columns: list):
        numeric_df = df[columns].apply(pd.to_numeric, errors='coerce').dropna()
        if not numeric_df.empty:
            return numeric_df.corr().to_dict()
        return {}

    @staticmethod
    def calculate_logistic_regression(df: pd.DataFrame, columns: list):
        results = {}
        if len(columns) < 2:
            print("Lojistik Hata: En az 2 sütun seçilmeli.")
            return results

        try:
            # 1. Veri Temizleme: Sayısal olmayanları NaN yap ve o satırları uçur
            temp_df = df[columns].apply(pd.to_numeric, errors='coerce').dropna()
            
            print(f"--- Lojistik Analiz Başladı ---")
            print(f"Orijinal Satır: {len(df)} -> Temizlenmiş Satır: {len(temp_df)}")

            if temp_df.empty:
                print("Lojistik Hata: Temizleme sonrası veri kalmadı (NaN veya Metin uyuşmazlığı).")
                return results

            # 2. Değişkenleri Ayır (Son sütun y, diğerleri X)
            y_col = columns[-1]
            X_cols = columns[:-1]
            
            y = temp_df[y_col]
            X = temp_df[X_cols]

            # 3. Kritik Kontrol: Bağımlı değişken sadece 0 ve 1 mi?
            unique_vals = sorted(y.unique().astype(int))
            if not np.array_equal(unique_vals, [0, 1]):
                print(f"Lojistik Hata: Bağımlı değişken ({y_col}) sadece 0 ve 1 içermeli. Bulunanlar: {unique_vals}")
                return results

            # 4. Model Kurulumu
            X = sm.add_constant(X) # Sabit terim (Intercept) ekle
            model = sm.Logit(y, X)
            res = model.fit(disp=0) # disp=0 konsolu kirletmemesini sağlar

            # 5. Sonuçları Paketle
            params = res.params
            p_values = res.pvalues
            # Odds Ratio (OR) hesapla: exp(B)
            odds_ratios = np.exp(params)

            for var in X_cols:
                results[var] = {
                    "katsayı": round(float(params[var]), 3),
                    "odds_ratio": round(float(odds_ratios[var]), 3),
                    "p_value": round(float(p_values[var]), 4)
                }
            print("Lojistik Analiz Başarıyla Tamamlandı.")

        except Exception as e:
            print(f"Lojistik Analiz Sırasında Matematiksel Hata: {str(e)}")
            # Matris tekil (singular) ise veya yakınsama olmazsa boş döner
        
        return results