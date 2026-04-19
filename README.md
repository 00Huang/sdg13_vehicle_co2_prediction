# SDG 13: Vehicle CO2 Emissions Prediction
## Personal Midterm Report — ML-based Solution for Climate Action

[English](#english) | [中文](#中文)

---

## English

> A Machine Learning-based solution addressing **UN Sustainable Development Goal 13 (Climate Action)** by predicting vehicle CO2 emissions from vehicle specifications.

### 📌 Midterm Report Progress

| Requirement | Status | Location |
|---|---|---|
| 1. Clear Problem Definition | ✅ Complete | [§1 Problem Definition](#1-problem-definition) |
| 2. Data Sources and Processing | ✅ Complete | [§2 Data & Processing](#2-data-sources-and-processing) · [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| 3. Model Selection and Discussion | ✅ Complete | [§3 Model Selection](#3-model-selection-and-discussion) |
| 4. Training and Testing | 🚧 In Progress | [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb) |

**First Delivery (Due 4/24, 30%)**: Requirements 1–3 complete with preliminary data processing and model exploration.

---

### 1. Problem Definition

**SDG Addressed**: SDG 13 — Climate Action

**Problem Statement**:
The transportation sector accounts for approximately 24% of global energy-related CO2 emissions, with road transport representing the majority share. Consumers purchasing vehicles often lack intuitive tools to assess how vehicle specifications (engine size, fuel type, fuel consumption) translate into CO2 emissions. Likewise, manufacturers and regulators need data-driven insights to identify high-impact design features and inform policy.

**Proposed Solution**:
Build regression models that predict vehicle CO2 emissions (g/km) from specification data, enabling:
- **Consumers**: Environmental decision support when purchasing vehicles
- **Manufacturers**: Identification of key design factors influencing emissions
- **Policymakers**: Data-driven evidence for carbon taxation and emission standards

**Why ML?**
Unlike rule-based approaches, ML models can capture nonlinear interactions between features (e.g., the combined effect of engine size and vehicle class) and provide feature importance rankings that reveal actionable insights.

---

### 2. Data Sources and Processing

#### 2.1 Data Source

- **Dataset**: [CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- **Author**: Debajyoti Podder (Kaggle)
- **Original Source**: [Government of Canada Open Data](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)
- **Size**: 7,385 records × 12 features
- **Target Variable**: `CO2 Emissions (g/km)`

#### 2.2 Feature Description

| Feature | Type | Description |
|---|---|---|
| Make, Model | Categorical | Vehicle brand and model |
| Vehicle Class | Categorical | SUV, compact, etc. |
| Engine Size (L) | Numerical | Displacement |
| Cylinders | Numerical | Number of cylinders |
| Transmission | Categorical | Transmission type |
| Fuel Type | Categorical | X/Z/D/E/N (gasoline/diesel/ethanol/natural gas) |
| Fuel Consumption (City/Hwy/Comb) | Numerical | L/100 km |
| **CO2 Emissions (g/km)** | Numerical | **Target** |

#### 2.3 Processing Pipeline

1. **Missing value inspection** — Check for and handle missing entries
2. **Feature renaming** — Clean column names for Python compatibility
3. **Multicollinearity removal** — Fuel_City / Fuel_Hwy / Fuel_Comb show correlation > 0.98; retain only Fuel_Comb
4. **Categorical encoding** — One-Hot Encoding for Make, Vehicle Class, Transmission, Fuel Type
5. **Numerical scaling** — StandardScaler for Engine Size, Cylinders, Fuel Consumption
6. **Outlier detection** — IQR method on target variable
7. **Train/test split** — 80/20 with `random_state=42`

#### 2.4 Sample Code Snippet

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv('data/raw/CO2 Emissions_Canada.csv')
df.columns = ['Make', 'Model', 'Vehicle_Class', 'Engine_Size', 'Cylinders',
              'Transmission', 'Fuel_Type', 'Fuel_City', 'Fuel_Hwy',
              'Fuel_Comb', 'Fuel_Comb_mpg', 'CO2_Emissions']

# Remove multicollinear and high-cardinality features
df_clean = df.drop(columns=['Model', 'Fuel_City', 'Fuel_Hwy', 'Fuel_Comb_mpg'])

# Define feature types
categorical_features = ['Make', 'Vehicle_Class', 'Transmission', 'Fuel_Type']
numerical_features = ['Engine_Size', 'Cylinders', 'Fuel_Comb']

# Build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split and transform
X = df_clean.drop(columns=['CO2_Emissions'])
y = df_clean['CO2_Emissions']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

Full implementation: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) and [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)

---

### 3. Model Selection and Discussion

Three regression models are compared, each representing a different ML paradigm:

| Model | Paradigm | Strengths | Limitations |
|---|---|---|---|
| **Linear Regression** | Parametric baseline | Highly interpretable; fast training; physically consistent with fuel-to-CO2 relationship | Cannot capture feature interactions or nonlinearities |
| **Random Forest Regressor** | Bagging ensemble | Captures nonlinear interactions; robust to outliers; provides feature importance | Lower interpretability; larger model size |
| **XGBoost Regressor** | Gradient boosting | State-of-the-art for tabular data; handles missing values; regularization support | More hyperparameters to tune; risk of overfitting |

#### Suitability Discussion

- **Linear Regression** serves as a theoretical baseline: burning one liter of gasoline produces ~2.3 kg of CO2, suggesting a near-linear relationship between fuel consumption and emissions. This makes it a meaningful reference point.
- **Random Forest** is well-suited for exploring **feature importance**, which directly supports the SDG 13 goal of identifying design factors with the greatest emission impact.
- **XGBoost** represents the current state-of-the-art for tabular regression and demonstrates the effort of learning a modern ML method — a key requirement of this midterm report.

**Critical Analysis (to be discussed in report)**:
Due to the near-linear relationship between fuel consumption and CO2, all models are expected to achieve high R² (>0.9). A secondary experiment will evaluate models **without the fuel consumption feature** to test whether vehicle specifications alone can predict emissions — a more practically meaningful scenario.

---

### 4. Training and Testing (In Progress)

Planned evaluation:
- **Metrics**: R², RMSE, MAE
- **Validation**: 5-fold cross-validation
- **Analysis**: Feature importance, residual plots, predicted vs. actual plots
- **Ablation**: Models with vs. without fuel consumption features

Progress tracked in [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb).

---

### 🗂️ Repository Structure

sdg13-vehicle-co2-prediction/
├── data/
│   ├── raw/              # Original CSV (not tracked in Git)
│   └── processed/        # Processed datasets
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Data Processing Pipeline
│   └── 03_modeling.ipynb         # Model Training & Evaluation
├── src/                  # Reusable Python modules
├── reports/
│   ├── figures/          # Exported plots
│   └── midterm_report.md # Formal report document
├── models/               # Saved trained models
├── requirements.txt
├── .gitignore
└── README.md

### 🛠️ Setup

```bash
git clone https://github.com/your-username/sdg13-vehicle-co2-prediction.git
cd sdg13-vehicle-co2-prediction

python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) and place `CO2 Emissions_Canada.csv` in `data/raw/`.

### 📚 References

- UN Sustainable Development Goals: [SDG 13 — Climate Action](https://sdgs.un.org/goals/goal13)
- Dataset: [Debajyoti Podder — CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- Original data: [Government of Canada Open Data](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)

### 👤 Author

[Your Name] — [Course Name], 2026

---
---

## 中文

> 以機器學習為基礎的解決方案，回應聯合國**永續發展目標 13（氣候行動）**，透過車輛規格預測 CO2 排放量。

### 📌 期中報告進度

| 作業要求 | 狀態 | 位置 |
|---|---|---|
| 1. 問題定義 | ✅ 完成 | [§1 問題定義](#1-問題定義) |
| 2. 資料來源與處理 | ✅ 完成 | [§2 資料與處理](#2-資料來源與處理) · [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| 3. 模型選擇與討論 | ✅ 完成 | [§3 模型選擇](#3-模型選擇與討論) |
| 4. 訓練與測試 | 🚧 進行中 | [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb) |

**第一次交件（4/24，30%）**：要求 1–3 完成，包含資料處理與新 ML 模型學習的實作成果。

---

### 1. 問題定義

**對應 SDG**：SDG 13 — 氣候行動

**問題陳述**：
運輸部門約占全球能源相關 CO2 排放量的 24%，其中道路運輸為主要來源。消費者購車時往往缺乏直觀工具來評估車輛規格（引擎大小、燃料類型、油耗）如何轉化為 CO2 排放量；車廠與政策制定者同樣需要資料驅動的洞察，以識別高影響力的設計因素並制定相關政策。

**解決方案**：
建立回歸模型，從車輛規格資料預測 CO2 排放量（g/km），應用包括：
- **消費者**：購車時的環保決策輔助
- **車廠**：識別影響排放的關鍵設計因素
- **政策制定者**：碳稅與排放標準的數據依據

**為何使用機器學習？**
相較於規則式方法，ML 模型能捕捉特徵間的非線性交互作用（例如引擎大小與車輛類型的綜合影響），並提供特徵重要性排名，揭示可行動的洞察。

---

### 2. 資料來源與處理

#### 2.1 資料來源

- **資料集**：[CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- **作者**：Debajyoti Podder（Kaggle）
- **原始來源**：[加拿大政府公開資料](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)
- **規模**：7,385 筆資料 × 12 個特徵
- **目標變數**：CO2 排放量（g/km）

#### 2.2 特徵說明

| 特徵 | 類型 | 說明 |
|---|---|---|
| Make, Model | 類別 | 車輛品牌與型號 |
| Vehicle Class | 類別 | SUV、小型車等 |
| Engine Size (L) | 數值 | 引擎排氣量 |
| Cylinders | 數值 | 汽缸數 |
| Transmission | 類別 | 變速箱類型 |
| Fuel Type | 類別 | X/Z/D/E/N（汽油/高辛烷汽油/柴油/乙醇/天然氣）|
| Fuel Consumption (City/Hwy/Comb) | 數值 | L/100 km |
| **CO2 Emissions (g/km)** | 數值 | **目標變數** |

#### 2.3 處理流程

1. **缺失值檢查** — 檢查並處理缺失資料
2. **欄位重新命名** — 整理欄位名稱以利 Python 處理
3. **移除多元共線特徵** — Fuel_City / Fuel_Hwy / Fuel_Comb 相關係數超過 0.98，僅保留 Fuel_Comb
4. **類別特徵編碼** — 對 Make、Vehicle Class、Transmission、Fuel Type 使用 One-Hot Encoding
5. **數值特徵標準化** — 對 Engine Size、Cylinders、Fuel Consumption 使用 StandardScaler
6. **異常值檢測** — 使用 IQR 方法檢查目標變數
7. **訓練/測試集切分** — 80/20，`random_state=42`

#### 2.4 程式碼範例

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 載入資料集
df = pd.read_csv('data/raw/CO2 Emissions_Canada.csv')
df.columns = ['Make', 'Model', 'Vehicle_Class', 'Engine_Size', 'Cylinders',
              'Transmission', 'Fuel_Type', 'Fuel_City', 'Fuel_Hwy',
              'Fuel_Comb', 'Fuel_Comb_mpg', 'CO2_Emissions']

# 移除多元共線特徵與高維類別特徵
df_clean = df.drop(columns=['Model', 'Fuel_City', 'Fuel_Hwy', 'Fuel_Comb_mpg'])

# 定義特徵類型
categorical_features = ['Make', 'Vehicle_Class', 'Transmission', 'Fuel_Type']
numerical_features = ['Engine_Size', 'Cylinders', 'Fuel_Comb']

# 建立前處理 pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 切分並轉換
X = df_clean.drop(columns=['CO2_Emissions'])
y = df_clean['CO2_Emissions']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

完整實作：[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) 與 [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb)

---

### 3. 模型選擇與討論

本專案比較三種不同機器學習範式的回歸模型：

| 模型 | 範式 | 優勢 | 限制 |
|---|---|---|---|
| **線性回歸** | 參數式基準模型 | 高可解釋性；訓練快速；符合油耗與 CO2 的物理線性關係 | 無法捕捉特徵交互作用與非線性 |
| **隨機森林** | Bagging 集成學習 | 能捕捉非線性交互作用；對異常值穩健；提供特徵重要性 | 可解釋性較低；模型較大 |
| **XGBoost** | 梯度提升 | 表格資料目前的 SOTA；可處理缺失值；支援正則化 | 超參數較多；過擬合風險 |

#### 適用性討論

- **線性回歸**作為理論基準：燃燒 1 公升汽油約產生 2.3 公斤 CO2，表示油耗與排放間接近線性關係，因此線性回歸是具物理意義的參考基準。
- **隨機森林**適合探索**特徵重要性**，直接回應 SDG 13 的核心目標——識別影響排放最大的設計因素。
- **XGBoost** 代表表格型回歸任務的當前最佳實踐，並展現學習現代 ML 方法的努力（期中報告的關鍵要求）。

**批判性分析（將於報告中深入討論）**：
由於油耗與 CO2 接近線性關係，所有模型預期均可達到高 R²（>0.9）。因此將進行補充實驗，評估**移除油耗特徵**後模型的表現——單純透過車輛規格是否能預測排放量，這是更具實用意義的情境。

---

### 4. 訓練與測試（進行中）

預計評估項目：
- **指標**：R²、RMSE、MAE
- **驗證方式**：5-fold 交叉驗證
- **分析**：特徵重要性、殘差圖、預測值 vs 實際值
- **消融實驗**：含油耗特徵 vs 不含油耗特徵

進度追蹤於 [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb)。

---

### 🗂️ 專案結構
```
sdg13-vehicle-co2-prediction/
├── data/
│   ├── raw/              # 原始 CSV（不上 Git）
│   └── processed/        # 處理後資料
├── notebooks/
│   ├── 01_eda.ipynb              # 探索性資料分析
│   ├── 02_preprocessing.ipynb    # 資料處理流程
│   └── 03_modeling.ipynb         # 模型訓練與評估
├── src/                  # 可重用 Python 模組
├── reports/
│   ├── figures/          # 匯出圖表
│   └── midterm_report.md # 正式報告文件
├── models/               # 已訓練模型
├── requirements.txt
├── .gitignore
└── README.md
```
### 🛠️ 環境建置

```bash
git clone https://github.com/your-username/sdg13-vehicle-co2-prediction.git
cd sdg13-vehicle-co2-prediction

python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

從 [Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) 下載資料集，將 `CO2 Emissions_Canada.csv` 放入 `data/raw/` 資料夾。

### 📚 參考資料

- 聯合國永續發展目標：[SDG 13 — 氣候行動](https://sdgs.un.org/goals/goal13)
- 資料集：[Debajyoti Podder — CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- 原始資料：[加拿大政府公開資料](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)

### 👤 作者

[你的名字] — [課程名稱], 2026