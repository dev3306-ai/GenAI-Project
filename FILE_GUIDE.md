# 📁 PROJECT FILE GUIDE

## Complete explanation of every file and its purpose

---

## 📂 PROJECT STRUCTURE

```
Solar Projet/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
├── FILE_GUIDE.md              # This file
│
├── models/
│   ├── __init__.py            # Package marker
│   └── forecaster.py          # ML models
│
└── utils/
    ├── __init__.py            # Package marker
    └── preprocessing.py       # Data processing
```

---

## 🎯 CORE APPLICATION FILES

### 1. `app.py` (Main Application)
**Purpose**: The heart of the project - Streamlit web interface

**What it does**:
- Creates the web UI using Streamlit
- Handles CSV file upload
- Orchestrates data flow: input → preprocessing → training → results
- Displays visualizations and metrics

**Key sections**:
```python
# 1. Data Input (lines 20-25)
- File uploader widget for CSV
- Success message on upload

# 2. Data Display (lines 27-35)
- Shows record count, features, date range
- Expandable table with raw data

# 3. Preprocessing (lines 37-45)
- Calls load_and_preprocess()
- Calls engineer_features()
- Shows preprocessing status

# 4. Model Configuration (lines 47-65)
- Target variable selection
- Feature selection (multiselect)
- Model type dropdown (RF, GB, Linear)
- Test size slider

# 5. Training (lines 67-90)
- Train-test split
- Feature scaling
- Model training
- Results storage in session_state

# 6. Results Display (lines 92-135)
- Metrics cards (MAE, RMSE, R²)
- Actual vs Predicted plot
- Feature importance chart
- Key insights
```

**Why we need it**: This is the user-facing interface for uploading and analyzing solar data.

---

### 2. `requirements.txt` (Dependencies)
**Purpose**: Lists all Python packages needed to run the project

**Contents**:
```
streamlit       # Web UI framework
pandas          # Data manipulation
numpy           # Numerical operations
scikit-learn    # Machine learning models
matplotlib      # Plotting (backup)
seaborn         # Statistical plots
plotly          # Interactive charts
```

**Why we need it**: 
- Ensures everyone has the same package versions
- Required for deployment (Streamlit Cloud, Hugging Face)
- One command install: `pip install -r requirements.txt`

---

### 3. `.gitignore` (Git Ignore Rules)
**Purpose**: Tells Git which files to ignore (not track)

**What it ignores**:
- `__pycache__/` - Python cache files
- `*.pyc` - Compiled Python files
- `.DS_Store` - Mac system files
- `venv/` - Virtual environment
- `*.csv` - Large data files
- `.streamlit/` - Streamlit config

**Why we need it**: Keeps repository clean, avoids committing unnecessary files

---

### 4. `README.md` (Project Documentation)
**Purpose**: Main documentation - first thing people see on GitHub

**Sections**:
1. **Project Overview**: What the project does
2. **Problem Statement**: Why it's needed
3. **System Architecture**: How it works (diagram)
4. **Dataset Description**: Input/output features
5. **Technical Stack**: Technologies used
6. **Installation**: How to set up
7. **Usage**: How to use the app
8. **Model Performance**: ML models explained
9. **Features**: What's implemented
10. **Project Structure**: File organization
11. **Team Contribution**: Who did what
12. **Methodology**: Technical approach

**Why we need it**: 
- Required for submission
- Helps evaluators understand the project
- Professional presentation

---

## 🤖 MACHINE LEARNING FILES

### 5. `models/forecaster.py` (ML Models)
**Purpose**: Contains all machine learning model logic

**Class**: `SolarForecaster`

**What it does**:
```python
# __init__(model_type)
- Creates model based on type: 'random_forest', 'gradient_boosting', 'linear_regression'
- Sets random_state=42 for reproducibility

# train(X_train, y_train)
- Trains the model on training data
- Uses scikit-learn's fit() method

# predict(X)
- Makes predictions on new data
- Returns predicted values

# evaluate(X_test, y_test)
- Calculates MAE (Mean Absolute Error)
- Calculates RMSE (Root Mean Squared Error)
- Calculates R² Score
- Returns dictionary with metrics and predictions

# get_feature_importance(feature_names)
- Returns feature importance for tree-based models
- Used for visualization
```

**Models explained**:

1. **Random Forest Regressor**
   - Ensemble of 100 decision trees
   - Each tree votes on prediction
   - Average of all votes = final prediction
   - Good for: Non-linear patterns, robust to outliers
   - Provides: Feature importance

2. **Gradient Boosting Regressor**
   - Builds trees sequentially
   - Each tree corrects previous errors
   - Good for: Highest accuracy, complex patterns
   - Slower but more accurate

3. **Linear Regression**
   - Simple linear model: y = mx + b
   - Good for: Baseline comparison, fast predictions
   - Assumes linear relationships

**Why we need it**: Encapsulates ML logic, makes it reusable and testable

---

### 6. `models/__init__.py` (Package Marker)
**Purpose**: Makes `models/` a Python package

**Content**: Just a comment (can be empty)

**Why we need it**: 
- Allows `from models.forecaster import SolarForecaster`
- Python requirement for package imports

---

## 🔧 UTILITY FILES

### 7. `utils/preprocessing.py` (Data Processing)
**Purpose**: Cleans and prepares data for ML models

**Functions**:

**1. `load_and_preprocess(df)`**
```python
What it does:
- Handles missing values using forward-fill and backward-fill
- Converts datetime column to datetime type
- Extracts temporal features:
  - hour (0-23): Time of day
  - day (1-31): Day of month
  - month (1-12): Month of year
  - day_of_week (0-6): Monday=0, Sunday=6

Why:
- Missing values break ML models
- Temporal features capture daily/seasonal patterns
- Solar power depends heavily on time
```

**2. `engineer_features(df, target_col)`**
```python
What it does:
- Creates rolling average (24-hour window)
- Smooths out noise in power generation
- Captures recent trends

Why:
- Rolling average helps model understand momentum
- Yesterday's generation predicts today's
- Improves model accuracy
```

**3. `scale_features(X_train, X_test)`**
```python
What it does:
- Normalizes features to mean=0, std=1
- Fits scaler on training data only
- Transforms both train and test data

Why:
- Features have different scales (temp: 0-40, irradiance: 0-1000)
- ML models work better with normalized data
- Prevents large values from dominating
```

**Why we need it**: Data quality determines model quality - garbage in, garbage out

---

### 8. `utils/__init__.py` (Package Marker)
**Purpose**: Makes `utils/` a Python package

**Why we need it**: Same as models/__init__.py - enables imports

---

## 📝 DOCUMENTATION FILES

### 9. `FILE_GUIDE.md` (This File)
**Purpose**: Explains every file in the project

**Why we need it**: 
- Helps team understand codebase
- Useful for viva preparation
- Makes project maintainable

---

## 🔄 DATA FLOW DIAGRAM

```
User Action (Upload CSV)
    ↓
app.py receives data
    ↓
utils/preprocessing.py
    ├─ load_and_preprocess()
    │   ├─ Handle missing values
    │   └─ Extract temporal features
    └─ engineer_features()
        └─ Create rolling averages
    ↓
User selects features & model
    ↓
app.py prepares data
    ├─ Train-test split
    └─ scale_features()
    ↓
models/forecaster.py
    ├─ Initialize model
    ├─ train()
    └─ evaluate()
    ↓
app.py displays results
    ├─ Metrics (MAE, RMSE, R²)
    ├─ Plots (Actual vs Predicted)
    └─ Feature Importance
```

---

## 🎯 WHY EACH COMPONENT EXISTS

### Why Streamlit (app.py)?
- **Fast development**: Create UI with pure Python
- **Interactive**: Widgets built-in (sliders, buttons, file upload)
- **ML-friendly**: Designed for data science apps
- **Free deployment**: Streamlit Cloud is free

### Why Scikit-Learn (forecaster.py)?
- **Industry standard**: Most popular ML library
- **Consistent API**: All models use fit/predict
- **Well-tested**: Reliable implementations
- **Easy to use**: Simple and intuitive

### Why Pandas (preprocessing.py)?
- **Data manipulation**: Best tool for tabular data
- **Time-series**: Excellent datetime handling
- **Missing values**: Built-in methods (ffill, bfill)
- **Integration**: Works seamlessly with scikit-learn

### Why Plotly (app.py)?
- **Interactive**: Zoom, pan, hover
- **Professional**: Publication-quality charts
- **Web-based**: Perfect for Streamlit
- **Responsive**: Adapts to screen size

---

## 📊 FILE IMPORTANCE RANKING

### Critical (Cannot run without):
1. ✅ `app.py` - Main application
2. ✅ `models/forecaster.py` - ML models
3. ✅ `utils/preprocessing.py` - Data processing
4. ✅ `requirements.txt` - Dependencies
5. ✅ `models/__init__.py` - Package marker
6. ✅ `utils/__init__.py` - Package marker

### Important (Needed for full functionality):
7. ✅ `README.md` - Documentation

### Optional (Nice to have):
8. ⭐ `.gitignore` - Clean repository
9. ⭐ `FILE_GUIDE.md` - Understanding

---

## 🎓 FOR VIVA PREPARATION

### Be ready to explain:

**app.py**:
- "This is the main Streamlit application that provides the web interface"
- "It handles CSV upload and orchestrates data flow from input to visualization"
- "Uses session_state to store model results"

**forecaster.py**:
- "Contains SolarForecaster class with 3 ML models"
- "Random Forest for non-linear patterns, Gradient Boosting for accuracy, Linear for baseline"
- "Provides train, predict, evaluate, and feature importance methods"

**preprocessing.py**:
- "Handles missing values with forward/backward fill"
- "Extracts temporal features (hour, month) for time-dependent patterns"
- "Scales features using StandardScaler for better model performance"

---

## 🚀 QUICK REFERENCE

### To run the app:
```bash
streamlit run app.py
```

### To test a component:
```python
# Test preprocessing
from utils.preprocessing import load_and_preprocess
df_processed = load_and_preprocess(df)

# Test model
from models.forecaster import SolarForecaster
model = SolarForecaster('random_forest')
model.train(X_train, y_train)
```

### CSV Format Required:
```
datetime,irradiance,temperature,cloud_cover,power_generated
2023-01-01 00:00:00,0.0,15.5,30.2,0.0
2023-01-01 01:00:00,0.0,14.8,25.1,0.0
...
```

---

## ✅ CHECKLIST FOR SUBMISSION

Files needed:
- [x] app.py
- [x] requirements.txt
- [x] models/forecaster.py
- [x] models/__init__.py
- [x] utils/preprocessing.py
- [x] utils/__init__.py
- [x] README.md
- [x] .gitignore
- [x] FILE_GUIDE.md

---

## 📞 SUMMARY

**Total Essential Files**: 7 code files + 2 documentation files = 9 files

**Lines of Code**:
- app.py: ~180 lines
- forecaster.py: ~50 lines
- preprocessing.py: ~40 lines
- **Total**: ~270 lines of actual code

**Why so organized?**:
- **Modularity**: Each file has one purpose
- **Maintainability**: Easy to update
- **Testability**: Can test each component
- **Professionalism**: Industry-standard structure

---

**This is a complete, production-ready ML application! 🎉**
