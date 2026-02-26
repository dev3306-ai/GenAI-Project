# ☀️ Intelligent Solar Energy Forecasting System

## Project Overview

An AI-driven analytics system for renewable energy management, specifically solar power generation forecasting. This project uses classical machine learning techniques to predict solar energy generation using historical power output, weather, and temporal data.

## 🎯 Problem Statement

Renewable energy sources like solar power are inherently variable due to weather conditions and time-dependent factors. Accurate forecasting of solar energy generation is crucial for:
- Grid stability and management
- Energy storage optimization
- Demand-supply balancing
- Cost reduction in energy distribution

This system addresses these challenges by providing accurate short-term and long-term solar energy generation forecasts.

## 🏗️ System Architecture

```
┌─────────────────┐
│   Data Input    │
│  (CSV/Sample)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Preprocessing           │
│ - Missing values        │
│ - Scaling               │
│ - Feature Engineering   │
└────────┬────────────────┘
         │
         ▼
┌────────────────────────┐
│  ML Models             │
│ - Random Forest        │
│ - Gradient Boost       │
│ - Linear Regression    │
└────────┬───────────────┘
         │
         ▼
┌─────────────────┐
│  Evaluation     │
│ - MAE/RMSE/R²   │
│ - Visualizations│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit UI    │
│ (Predictions)   │
└─────────────────┘
```

## 📊 Dataset Description

### Input Features:
- **datetime**: Timestamp of measurement
- **irradiance**: Solar irradiance (W/m²)
- **temperature**: Ambient temperature (°C)
- **cloud_cover**: Cloud coverage percentage (%)

### Engineered Features:
- **hour**: Hour of day (0-23)
- **day**: Day of month
- **month**: Month of year
- **day_of_week**: Day of week (0-6)
- **power_rolling_24h**: 24-hour rolling average of power generation

### Target Variable:
- **power_generated**: Solar power output (kW)

## 🔧 Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: Scikit-Learn
- **UI Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd "Solar Projet"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 💻 Usage

### Option 1: Upload Your Data
1. Click "Upload CSV" in the sidebar
2. Upload a CSV file with columns: datetime, irradiance, temperature, cloud_cover, power_generated
3. Configure model settings
4. Click "Train Model"

### Option 2: Use Sample Data
1. Select "Use Sample Data" in the sidebar
2. Choose duration (30-365 days)
3. Click "Generate Sample Data"
4. Configure model settings
5. Click "Train Model"

## 📈 Model Performance

The system supports three regression models:

1. **Random Forest Regressor**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Provides feature importance

2. **Gradient Boosting Regressor**
   - Sequential ensemble method
   - High accuracy
   - Good for complex patterns

3. **Linear Regression**
   - Baseline model
   - Fast training
   - Interpretable coefficients

### Evaluation Metrics:
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **R² Score**: Proportion of variance explained

## 🎨 Features

### ✅ Milestone 1 (Mid-Sem) Features:
- [x] Data upload and preprocessing
- [x] Missing value handling
- [x] Feature scaling (StandardScaler)
- [x] Temporal feature extraction
- [x] Feature engineering (rolling averages)
- [x] Multiple ML models (RF, GB, Linear)
- [x] Model training pipeline
- [x] Performance evaluation (MAE, RMSE, R²)
- [x] Interactive visualizations
- [x] Prediction vs Actual plots
- [x] Feature importance analysis
- [x] Streamlit UI
- [x] Sample data generation

## 📁 Project Structure

```
Solar Projet/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── models/
│   └── forecaster.py          # ML model implementations
├── utils/
│   ├── preprocessing.py       # Data preprocessing utilities
│   └── data_generator.py      # Sample data generator
├── data/                      # Data directory (optional)
└── README.md                  # Project documentation
```

## 👥 Team Contribution

| Member | Contribution |
|--------|-------------|
| Member 1 | Data preprocessing, feature engineering |
| Member 2 | ML model implementation, evaluation |
| Member 3 | Streamlit UI, visualizations |
| Member 4 | Documentation, testing |

## 🔬 Methodology

### 1. Data Preprocessing
- Forward-fill and backward-fill for missing values
- StandardScaler for feature normalization
- Temporal feature extraction from datetime

### 2. Feature Engineering
- Hour, day, month, day_of_week extraction
- Rolling averages for trend capture
- Interaction features between weather variables

### 3. Model Training
- Train-test split (80-20 default)
- Scikit-learn pipeline integration
- Cross-validation ready architecture

### 4. Evaluation
- Multiple metrics (MAE, RMSE, R²)
- Visual comparison of predictions vs actuals
- Feature importance analysis for tree-based models

## 🎯 Future Enhancements (Milestone 2)

- [ ] Agentic AI integration with LangGraph
- [ ] RAG-based grid management recommendations
- [ ] Risk analysis and variability assessment
- [ ] Energy utilization strategy generation
- [ ] Grid balancing recommendations

## 📝 License

This project is developed as part of the GenAI course capstone project.

## 🔗 Links

- **Live Demo**: [To be deployed]
- **GitHub Repository**: [Your repo URL]
- **Demo Video**: [To be uploaded]

## 📞 Contact

For questions or issues, please contact the team members or raise an issue in the repository.

---

**Version**: 1.0 (Milestone 1)  
**Last Updated**: [Current Date]
