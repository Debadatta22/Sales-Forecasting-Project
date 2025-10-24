# Sales-Forecasting-Project

# 🎯 Sales Forecasting - Retail Analytics Project

<div align="center">

**Made by Debadatta Rout**  
*Internship at Acmegrade in Data Science*  
*Batch - DS Aug 2025*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM-green)](https://lightgbm.readthedocs.io)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

</div>

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Detailed Implementation](#-detailed-implementation)
- [Results & Performance](#-results--performance)
- [Key Insights](#-key-insights)
- [Conclusion](#-conclusion)
- [How to Run](#-how-to-run)
- [Future Enhancements](#-future-enhancements)

## 🎯 Project Overview

This project implements a comprehensive **Sales Forecasting System** for retail analytics using machine learning. The system predicts item outlet sales based on various product and outlet characteristics, enabling businesses to optimize inventory management, marketing strategies, and operational efficiency.

**Project Duration**: 4 Weeks  
**Technologies Used**: Python, Pandas, Scikit-learn, LightGBM, XGBoost, Matplotlib, Seaborn  
**Project Type**: Supervised Learning - Regression

## 💼 Business Problem

Retail businesses face significant challenges in inventory management and sales prediction. Inaccurate sales forecasts lead to:

- **Overstocking**: Increased holding costs and potential wastage
- **Understocking**: Lost sales opportunities and customer dissatisfaction  
- **Inefficient Marketing**: Poor allocation of promotional budgets
- **Operational Inefficiencies**: Suboptimal staff scheduling and resource allocation

**Solution**: Develop a machine learning model that accurately predicts item-level sales across different outlets, enabling data-driven decision making.

## 📊 Dataset Description

The dataset contains **8,523 records** with **12 initial features** describing products and outlet characteristics:

### Original Features:
- **Product Attributes**: Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP
- **Outlet Attributes**: Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type
- **Target Variable**: Item_Outlet_Sales

### Data Quality Assessment:
- **Total Records**: 8,523
- **Missing Values**: Item_Weight (1,463), Outlet_Size (2,410)
- **Complete Records**: 4,650 (54.56%)
- **Duplicate Records**: 0

## 🔄 Project Workflow

<img width="3576" height="8646" alt="image" src="https://github.com/user-attachments/assets/8af72772-a5aa-49f0-8475-fa4cb018ff61" />




## ⚙️ Detailed Implementation

### Step 1: Environment Setup and Library Installation
**1.1** Installation of essential Python libraries including LightGBM, XGBoost, and data science stack
**1.2** Configuration of visualization settings and warning filters
**1.3** Setting up the project directory structure

### Step 2: Data Loading and Initial Exploration
**2.1** Loading the Train.csv dataset into pandas DataFrame
**2.2** Basic dataset information analysis (shape, memory usage, data types)
**2.3** Initial data quality assessment and missing values identification

### Step 3: Comprehensive Data Analysis
**3.1** Statistical summary of numerical and categorical variables
**3.2** Distribution analysis of target variable (Item_Outlet_Sales)
**3.3** Identification of data inconsistencies and outliers
**3.4** Creation of data quality report with completeness metrics

### Step 4: Missing Values Analysis
**4.1** Detailed examination of missing values pattern in Item_Weight and Outlet_Size
**4.2** Analysis of relationships between missing values and other variables
**4.3** Development of imputation strategy based on data patterns

### Step 5: Data Imputation
**5.1** Item_Weight imputation using mean values grouped by Item_Identifier
**5.2** Outlet_Size imputation using mode values grouped by Outlet_Type
**5.3** Verification of imputation success and data completeness
**5.4** Creation of data backup before and after imputation

### Step 6: Data Cleaning and Feature Engineering
**6.1** Standardization of categorical values (Item_Fat_Content)
**6.2** Handling zero values in Item_Visibility by replacing with mean
**6.3** Creation of new features:
- **New_Item_Type**: Categorization based on Item_Identifier prefixes
- **Outlet_Years**: Calculation of outlet age from establishment year
- **Item_Category**: Grouping similar item types for better generalization

**6.4** Business rule implementation: Setting Non-Consumable items to 'Non-Edible' fat content

### Step 7: Exploratory Data Analysis and Visualization
**7.1** Distribution analysis of all numerical variables with statistical annotations
**7.2** Categorical variable analysis using count plots and bar charts
**7.3** Sales analysis by different categories (Outlet Type, Item Category)
**7.4** Correlation analysis and heatmap visualization
**7.5** Time-based analysis of outlet establishment patterns
**7.6** Box plot analysis for sales distribution across different segments

### Step 8: Data Preprocessing for Machine Learning
**8.1** Identification and separation of numerical and categorical features
**8.2** Label encoding for ordinal variables (Outlet_Size, Outlet_Location_Type, Item_Fat_Content)
**8.3** One-hot encoding for nominal variables (Outlet_Type, New_Item_Type, Item_Category)
**8.4** Removal of identifier columns not useful for modeling
**8.5** Creation of final feature set with 22 engineered features
**8.6** Target variable analysis and transformation verification

### Step 9: Model Training and Evaluation
**9.1** Data splitting into training (70%) and testing (30%) sets
**9.2** Implementation of 8 machine learning algorithms:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- LightGBM Regressor
- XGBoost Regressor

**9.3** Comprehensive model evaluation using multiple metrics:
- R² Score (Coefficient of Determination)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Cross-validation scores

**9.4** Model performance comparison and visualization
**9.5** Identification of best performing model (LightGBM)

### Step 10: Hyperparameter Tuning
**10.1** Definition of parameter search space for LightGBM
**10.2** Implementation of RandomizedSearchCV with 20 iterations
**10.3** Cross-validation with 3 folds for robust performance estimation
**10.4** Performance comparison between original and tuned models
**10.5** Feature importance analysis of tuned model

### Step 11: Model Diagnostics and Validation
**11.1** Residual analysis and assumption checking
**11.2** Actual vs Predicted values visualization
**11.3** Error distribution analysis
**11.4** Business interpretation of model results

### Step 12: Interactive Prediction System
**12.1** Development of user-friendly input interface
**12.2** Real-time data preprocessing pipeline
**12.3** Prediction output with confidence intervals
**12.4** Business insights and recommendations generation
**12.5** Model serialization and deployment preparation

## 📈 Results & Performance

### Model Performance Comparison:

| Model | R² Score | RMSE | MAE | Status |
|-------|----------|------|-----|---------|
| LightGBM (Tuned) | 0.5812 | 1083.13 | 752.04 | 🏆 **Best** |
| Lasso Regression | 0.5688 | 1098.98 | 809.58 | ✅ Good |
| Ridge Regression | 0.5683 | 1099.65 | 810.41 | ✅ Good |
| Linear Regression | 0.5683 | 1099.68 | 810.50 | ✅ Good |
| Random Forest | 0.5497 | 1123.06 | 780.54 | ⚠️ Average |
| XGBoost | 0.5307 | 1146.55 | 803.64 | ⚠️ Average |
| Extra Trees | 0.5176 | 1162.45 | 813.95 | ❌ Poor |
| Decision Tree | 0.2015 | 1495.57 | 1040.90 | ❌ Poor |

### Key Performance Metrics:
- **Best Model**: LightGBM (Gradient Boosting)
- **R² Score**: 0.5812 (58.12% variance explained)
- **RMSE**: 1,083.13 (Root Mean Square Error)
- **MAE**: 752.04 (Mean Absolute Error)
- **Improvement over Baseline**: 42.3% better than mean prediction

## 🔍 Key Insights

### Top 5 Features Driving Sales Predictions:
1. **Item_MRP** (24.8%) - Product maximum retail price is the strongest predictor
2. **Outlet_Type_Supermarket Type1** (18.3%) - Specific outlet type characteristics
3. **Outlet_Type_Grocery Store** (12.1%) - Grocery stores show distinct patterns
4. **Outlet_Years** (9.7%) - Older outlets have different sales behaviors
5. **Item_Visibility** (8.4%) - Product visibility significantly impacts sales

### Business Insights:
- **Pricing Strategy**: Item MRP is the most important factor, suggesting pricing optimization opportunities
- **Outlet Segmentation**: Different outlet types require customized sales strategies
- **Product Placement**: Item visibility strongly influences sales, highlighting display importance
- **Outlet Age**: Older establishments show predictable sales patterns for inventory planning

## 🎯 Conclusion

### Project Achievements:
✅ **Successfully developed** a robust sales forecasting system with 58.12% variance explanation  
✅ **Implemented comprehensive** data preprocessing pipeline handling real-world data challenges  
✅ **Evaluated multiple algorithms** and selected optimal model through rigorous testing  
✅ **Created interactive system** for real-time sales predictions with business insights  
✅ **Delivered actionable insights** for retail business optimization  

### Technical Accomplishments:
- Handled complex missing data patterns using intelligent imputation strategies
- Engineered meaningful features that improved model performance
- Achieved significant improvement over baseline models
- Built a scalable system ready for deployment and further enhancement

### Business Impact:
The developed system enables retailers to:
- Make data-driven inventory decisions
- Optimize product placement and pricing strategies
- Improve operational efficiency through accurate forecasting
- Enhance customer satisfaction by maintaining optimal stock levels

## 🚀 How to Run

### Prerequisites:
```bash
Python 3.8+
Jupyter Notebook
Required libraries: pandas, numpy, scikit-learn, lightgbm, xgboost, matplotlib, seaborn

```

## Installation:

```
# Clone the repository
git clone <repository-url>

# Install required packages
pip install -r requirements.txt

# Or install individually
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

## Execution:

Open Sales_Forecasting_Project.ipynb in Jupyter Notebook

Run cells sequentially from top to bottom

For interactive predictions, run the prediction system cell

Follow the prompts to input product and outlet details

## 🔮 Future Enhancements
Short-term Improvements:
Integration with real-time data streams

Additional feature engineering (seasonality, promotions)

Ensemble methods combining multiple models

Automated hyperparameter optimization

## Long-term Vision:
Web application deployment with REST API

Integration with inventory management systems

Real-time model retraining and monitoring

Multi-store chain optimization features

Demand forecasting for new product launches

## Advanced Features:
Time-series analysis for seasonal patterns

Customer segmentation integration

Competitor pricing data incorporation

Economic indicator integration
