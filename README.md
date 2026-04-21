# SDG 13: Vehicle CO2 Emissions Prediction

> Using machine learning to predict vehicle CO2 emissions from specifications — and challenging the long-held policy assumption that "bigger engines pollute more."

[English](#english) | [中文](#中文)

---

## English

### 🔑 Key Findings

- **Fuel consumption is the dominant, linear lever**: each 1 L/100km reduction → ~17 g/km less CO2 (≈ 510 kg saved annually at 15,000 km/yr)
- **Engine size and cylinder count contribute < 1%** to emission prediction — challenging displacement-based taxation systems (e.g., Japan's *kei-car*, EU's CO2/kW scheme)
- **Ethanol vehicles exhibit a counterintuitive double-effect**: lower per-liter carbon coefficient, but market availability clusters at high fuel consumption (all ≥ 17 L/100km), canceling the fuel advantage
- **Best model (XGBoost, hyperparameter-tuned)**: R² = 0.9982, RMSE = 2.52 g/km on held-out test set

---

### 1. Problem Definition

**SDG Addressed**: SDG 13 — Climate Action

Road transportation accounts for roughly a quarter of global energy-related CO2 emissions. Most current vehicle tax systems and eco-labels grade by **engine displacement**, built on the intuition that larger engines emit more. Yet this intuition has rarely been **quantitatively tested** against empirical data.

This project addresses three gaps:

1. **Consumers** choosing "eco-friendly" vehicles often rely on engine size as a proxy, but lack data-driven tools to assess the actual emission impact of different specifications.
2. **Manufacturers** need to identify which design decisions most affect emissions — beyond simple heuristics.
3. **Policymakers** designing carbon taxation need empirical evidence for which vehicle features best predict emissions.

**Solution**: Build a regression model to predict CO2 emissions (g/km) from vehicle specifications, and use **interpretability analysis (SHAP)** to quantify each feature's contribution — providing actionable, evidence-based insights for all three audiences.

**Why ML, not simple formulas?** Although burning 1 L of gasoline produces approximately 2.3 kg of CO2 (a near-linear chemistry), real vehicle emissions depend on **interactions** between fuel type, fuel consumption, engine design, and transmission. ML models can capture these non-linear interactions while providing interpretable per-feature contributions.

---

### 2. Data Sources and Processing

**Data source**: Canadian government open data on new vehicle registrations (2014–2020), published via [Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) — **7,385 vehicles × 12 features**. Target variable: `CO2 Emissions (g/km)`, range 96–522.

**Key preprocessing decisions** (detailed rationale in notebooks):

| Step | Action | Rationale |
|---|---|---|
| 1 | Dropped `Fuel_City`, `Fuel_Hwy`, `Fuel_Comb_mpg` | Multicollinearity with `Fuel_Comb` (r > 0.95) |
| 2 | Dropped `Model`, `Make` | High cardinality (2000+ / 40+ unique values) with long-tail distribution |
| 3 | Removed 1 row where `Fuel_Type = 'N'` | Insufficient samples to learn this category |
| 4 | One-Hot encoded remaining categorical features | No inherent ordinal relationship |
| 5 | StandardScaler for numerical features | Different scales (Engine_Size: 0.9–8.4 vs Fuel_Comb: 4–26) |
| 6 | 80/20 train/test split with `random_state=42` | Reproducibility; stratified splitting not applicable for regression |

All transformations are **fit on training data only** and applied to test data via `transform()`, preventing data leakage. The fitted pipeline is serialized to `models/preprocessor.joblib` for reuse.

📓 *Full implementation with sample code and EDA visualizations*: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) and [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)

---

### 3. Model Selection and Discussion

Three models spanning different ML paradigms were compared:

**Linear Regression** (baseline) — Chosen as a physically meaningful reference point. Since CO2 emissions follow a near-linear relationship with fuel consumption (~2.3 kg CO2 per liter of gasoline), linear regression establishes whether the problem needs anything more sophisticated.

**Random Forest Regressor** (bagging ensemble) — Chosen for its ability to capture feature interactions (e.g., different slopes for different fuel types, as revealed in EDA scatter plots) and for providing built-in feature importance rankings.

**XGBoost Regressor** (gradient boosting) — Chosen as the current state-of-the-art for tabular regression, and to fulfill the assignment requirement of learning a modern ML method. Its sequential boosting architecture tends to outperform bagging on complex tabular data.

**Comparison strategy**: Models are evaluated on a held-out test set using R², RMSE, and MAE (classification metrics like accuracy and confusion matrix do not apply to regression). XGBoost is additionally tuned via 5-fold cross-validated **Grid Search** over 27 hyperparameter configurations.

**Interpretability layer**: Beyond comparing predictive accuracy, this project uses **SHAP (SHapley Additive exPlanations)** to analyze *how* each feature contributes to predictions. This distinguishes features that merely correlate with emissions from those that have quantifiable, isolated effects — critical for drawing reliable policy conclusions.

📓 *Full model training, tuning, and SHAP analysis*: [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb)

---

### 🗺️ How to Navigate This Repository

| File | What You'll See |
|---|---|
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Data exploration, correlation analysis, fuel-type stratification discovery |
| [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) | Preprocessing pipeline with decision rationale |
| [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Four-model comparison, Grid Search, SHAP interpretability |

### 📋 Assignment Requirement Mapping

| Requirement | Location |
|---|---|
| 1. Problem Definition | [Section 1](#1-problem-definition) |
| 2. Data Sources and Processing (incl. code sample) | [Section 2](#2-data-sources-and-processing) · [`01_eda.ipynb`](notebooks/01_eda.ipynb) · [`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) |
| 3. Model Selection and Discussion | [Section 3](#3-model-selection-and-discussion) · [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) |
| 4. Training and Testing | [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) |

### 🛠️ Reproducing This Project

```bash
git clone https://github.com/00Huang/sdg13_vehicle_co2_prediction.git
cd sdg13_vehicle_co2_prediction

python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

Download `CO2 Emissions_Canada.csv` from the Kaggle link above and place it in `data/raw/`.

### 📚 References

- UN SDG 13 — [Climate Action](https://sdgs.un.org/goals/goal13)
- Dataset: [Debajyoti Podder — CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- Original data: [Government of Canada Open Data](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)

### 👤 Author

[黃柄博] — [MLSECOPS], 2026

---

---

## 中文

### 🔑 核心發現

- **油耗是主導的線性減碳槓桿**：每降低 1 L/100km → 減排約 17 g/km（以每年 15,000 km 計算，年省約 510 kg CO2）
- **引擎排氣量與汽缸數對排放預測的貢獻 < 1%**——挑戰現行以排氣量分級的稅制（日本輕自動車、歐盟 CO2/kW 方案）
- **乙醇車呈現反直覺的雙重效應**：每公升碳排係數較低，但市場上的乙醇車油耗皆偏高（全數 ≥ 17 L/100km），燃料優勢被車款設計抵銷
- **最佳模型（調參後 XGBoost）**：測試集 R² = 0.9982、RMSE = 2.52 g/km

---

### 1. 問題定義

**對應 SDG**：SDG 13 — 氣候行動

道路運輸約占全球能源相關 CO2 排放的四分之一。目前多數國家的汽車稅制與環保標籤以**引擎排氣量**為分級基礎，預設「引擎越大越耗能」。然而這個直覺很少被**實證資料量化檢驗**。

本專案處理三個面向：

1. **消費者**選購「環保車」時，常以引擎大小為替代指標，但缺乏資料驅動的工具評估不同規格對實際排放的影響。
2. **車廠**需要識別哪些設計決策最影響排放——而非只靠經驗法則。
3. **政策制定者**設計碳稅時，需要實證依據來判斷哪些車輛特徵最能預測排放。

**解決方案**：建立回歸模型從車輛規格預測 CO2 排放量（g/km），並透過 **SHAP 解釋性分析**量化每個特徵的貢獻——為三類讀者提供可行動的證據基礎。

**為何使用 ML 而非簡單公式？** 雖然燃燒 1 公升汽油約產生 2.3 公斤 CO2（接近線性的化學反應），但真實的車輛排放取決於**燃料類型、油耗、引擎設計、變速箱之間的交互作用**。ML 模型能捕捉這些非線性交互作用，同時提供可解釋的特徵貢獻度。

---

### 2. 資料來源與處理

**資料來源**：加拿大政府 2014–2020 新車登錄公開資料，透過 [Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) 取得——**7,385 筆車輛 × 12 個特徵**。目標變數：`CO2 Emissions (g/km)`，範圍 96–522。

**關鍵前處理決策**（詳細理由見 notebook）：

| 步驟 | 處理方式 | 理由 |
|---|---|---|
| 1 | 移除 `Fuel_City`、`Fuel_Hwy`、`Fuel_Comb_mpg` | 與 `Fuel_Comb` 多元共線（r > 0.95）|
| 2 | 移除 `Model`、`Make` | 高基數（2000+ / 40+ 唯一值）且呈長尾分布 |
| 3 | 移除 `Fuel_Type = 'N'` 的 1 筆資料 | 樣本不足以學習該類別 |
| 4 | 剩餘類別特徵採用 One-Hot 編碼 | 無內在順序關係 |
| 5 | 數值特徵採用 StandardScaler | 尺度差異大（Engine_Size: 0.9–8.4 vs Fuel_Comb: 4–26）|
| 6 | 80/20 訓練/測試集切分（`random_state=42`）| 確保可重現；回歸任務不適用 stratified 切分 |

所有轉換**僅在訓練集上 fit**，測試集透過 `transform()` 套用，避免資料洩漏。訓練好的 pipeline 序列化儲存至 `models/preprocessor.joblib` 供重複使用。

📓 *完整實作與程式碼範例、EDA 視覺化*：[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) 與 [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)

---

### 3. 模型選擇與討論

比較三種代表不同 ML 範式的模型：

**Linear Regression（線性回歸，基準模型）**——作為具物理意義的參考點。由於 CO2 排放與油耗呈接近線性的關係（燃燒 1 公升汽油約產生 2.3 公斤 CO2），線性回歸能檢驗這個問題是否需要更複雜的模型。

**Random Forest Regressor（隨機森林，Bagging 集成）**——能捕捉特徵交互作用（例如 EDA 散點圖揭露的不同燃料類型有不同斜率），並提供內建的特徵重要性排名。

**XGBoost Regressor（梯度提升）**——表格資料目前的 SOTA，同時滿足作業要求「學習現代 ML 方法」。其序列式提升架構在複雜表格資料上通常優於 Bagging。

**比較策略**：使用 R²、RMSE、MAE 在獨立測試集上評估（分類指標如準確率、混淆矩陣不適用於回歸任務）。XGBoost 額外透過 5-fold 交叉驗證的 **Grid Search** 搜尋 27 組超參數組合進行調整。

**解釋性分析**：除了比較預測準確度，本專案使用 **SHAP（SHapley Additive exPlanations）**分析每個特徵**如何**貢獻預測值。這能區分「僅與排放相關」的特徵和「有可量化獨立效應」的特徵——這對於導出可信的政策結論至關重要。

📓 *完整模型訓練、調參與 SHAP 分析*：[`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb)

---

### 🗺️ 如何瀏覽這個 Repository

| 檔案 | 看什麼 |
|---|---|
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | 資料探索、相關性分析、燃料類型分層現象的發現 |
| [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) | 前處理 pipeline 與決策理由 |
| [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb) | 四模型比較、Grid Search、SHAP 解釋性分析 |

### 📋 作業 Requirement 對應

| 作業要求 | 位置 |
|---|---|
| 1. 問題定義 | [第 1 節](#1-問題定義) |
| 2. 資料來源與處理（含程式碼範例） | [第 2 節](#2-資料來源與處理) · [`01_eda.ipynb`](notebooks/01_eda.ipynb) · [`02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) |
| 3. 模型選擇與討論 | [第 3 節](#3-模型選擇與討論) · [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) |
| 4. 訓練與測試 | [`03_modeling.ipynb`](notebooks/03_modeling.ipynb) |

### 🛠️ 環境重建

```bash
git clone https://github.com/00Huang/sdg13_vehicle_co2_prediction.git
cd sdg13_vehicle_co2_prediction

python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

從上述 Kaggle 連結下載 `CO2 Emissions_Canada.csv`，放入 `data/raw/`。

### 📚 參考資料

- 聯合國 SDG 13 — [氣候行動](https://sdgs.un.org/goals/goal13)
- 資料集：[Debajyoti Podder — CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- 原始資料：[加拿大政府公開資料](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)

### 👤 作者

[黃柄博] — [人工智慧開發與安全], 2026