import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

class StatsService:
    @staticmethod
    def calculate_descriptive(df: pd.DataFrame, columns: list):
        results = {}
        for col in columns:
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
        try:
            temp_df = df[columns].apply(pd.to_numeric, errors='coerce').dropna()
            if temp_df.empty: return results
            y_col, X_cols = columns[-1], columns[:-1]
            y, X = temp_df[y_col], temp_df[X_cols]
            if not np.array_equal(sorted(y.unique().astype(int)), [0, 1]): return results
            X = sm.add_constant(X)
            model = sm.Logit(y, X)
            res = model.fit(disp=0)
            params, p_values, odds_ratios = res.params, res.pvalues, np.exp(res.params)
            for var in X_cols:
                results[var] = {
                    "katsayı": round(float(params[var]), 3),
                    "odds_ratio": round(float(odds_ratios[var]), 3),
                    "p_value": round(float(p_values[var]), 4)
                }
        except Exception as e: print(f"Lojistik Hata: {e}")
        return results

    @staticmethod
    def calculate_chi_square(df: pd.DataFrame, columns: list):
        results = {}
        try:
            if len(columns) < 2: return results
            c1, c2 = columns[0], columns[1]
            clean_df = df[[c1, c2]].dropna()
            ct = pd.crosstab(clean_df[c1], clean_df[c2])
            chi2, p, dof, ex = stats.chi2_contingency(ct)
            results[f"{c1} & {c2}"] = {"Chi2": round(chi2, 2), "p_değeri": round(p, 4), "Sonuç": "Anlamlı" if p < 0.05 else "Anlamsız"}
        except Exception as e: print(f"Chi2 Hata: {e}")
        return results

    @staticmethod
    def calculate_anova(df: pd.DataFrame, columns: list):
        results = {}
        try:
            if len(columns) < 2: return results
            g_col, v_col = columns[0], columns[1]
            clean_df = df[[g_col, v_col]].dropna()
            groups = [clean_df[clean_df[g_col] == g][v_col] for g in clean_df[g_col].unique()]
            f_stat, p = stats.f_oneway(*groups)
            results[v_col] = {"F-Stat": round(f_stat, 2), "p_değeri": round(p, 4), "Sonuç": "Anlamlı" if p < 0.05 else "Anlamsız"}
        except Exception as e: print(f"ANOVA Hata: {e}")
        return results

    @staticmethod
    def calculate_mann_whitney(df: pd.DataFrame, columns: list):
        results = {}
        try:
            if len(columns) < 2: return results
            s1 = pd.to_numeric(df[columns[0]], errors='coerce').dropna()
            s2 = pd.to_numeric(df[columns[1]], errors='coerce').dropna()
            u, p = stats.mannwhitneyu(s1, s2)
            results[f"{columns[0]} vs {columns[1]}"] = {"U-Stat": round(u, 2), "p_değeri": round(p, 4), "Sonuç": "Anlamlı" if p < 0.05 else "Anlamsız"}
        except Exception as e: print(f"MWU Hata: {e}")
        return results