# Real Estate Investment Advisor

A Python and Streamlit project for exploring property listings,
classifying relative affordability, and demonstrating five-year
property-price projections.

# Real Estate Investment Advisor

🌐 **Live App:** [Open the app](https://estatescope-real-estate-advisor.streamlit.app/)

💻 **GitHub Repository:** [View the code](https://github.com/Shannu1711/Real-Estate-Investment-Advisor)

## Project objectives

1. Classify whether a property meets a defined “Good Investment” rule.
2. Predict a five-year price target generated using an assumed growth rate.
3. Provide property filters, charts, model scores and coefficient explanations.

## Important interpretation

“Good Investment” is a constructed affordability label, not evidence
of actual profitability.

Five-year prices assume 8% annual growth. The dataset contains no
observed future prices or realised investment returns. Model scores
therefore measure agreement with the constructed targets, not
real-world investment forecasting accuracy.

## Dataset

The supplied india_housing_prices.csv contains 250,000 properties
and 23 original columns.

Features include location, property type, BHK, size, current price,
property age, furnishing, nearby facilities, transport access,
parking, security and amenities.

Property ages imply a dataset reference year of 2025.

## Data preparation

- Checked missing values, blank text, duplicate IDs and duplicate records.
- Trimmed leading and trailing spaces from text.
- Recalculated price per square foot in rupees:
  Price_in_Lakhs × 100000 / Size_in_SqFt.
- Excluded Floor_No and Total_Floors because 116,304 records
  (46.52%) had a floor number higher than total floors.
- Retained all 250,000 property rows.
- Created five binary amenity indicators and Amenities_Count.
- Retained statistically flagged price-per-square-foot outliers
  because the flags alone did not establish data errors.

## Exploratory analysis

The notebook includes price and size distributions, city and state
comparisons, locality summaries, BHK distributions, correlations,
and comparisons involving amenities and nearby facilities.

Most examined characteristics show little association with price
in this dataset. These findings should not be generalized to the
entire Indian property market.

## Target definitions

### Classification

Good_Investment = 1 when price per square foot is at or below the
training-set median for the same state, city and property type.

An overall training median is the fallback for an unseen group.

### Regression

Future_Price_5Y_Lakhs = Price_in_Lakhs × (1.08 ** 5)

The 8% rate is an illustrative project assumption.

## Modelling

- Training: 175,000 properties (70%).
- Validation: 37,500 properties (15%).
- Test: 37,500 properties (15%).
- Random seed: 42.
- Numerical preprocessing: median imputation and standard scaling.
- Categorical preprocessing: most-frequent imputation and one-hot encoding.
- Preprocessing and affordability benchmarks were fitted using training data.
- Models were selected using validation results.

Classification candidates:
- Majority baseline
- Logistic Regression
- Random Forest Classifier

Regression comparisons:
- Mean baseline
- Direct assumed-growth formula
- Linear Regression
- Random Forest Regressor

## Selected models and test results

| Task | Selected model | Metric | Result |
|---|---|---|---:|
| Classification | Logistic Regression | Accuracy | 99.1467% |
| Classification | Logistic Regression | Precision | 0.991062 |
| Classification | Logistic Regression | Recall | 0.991907 |
| Classification | Logistic Regression | F1 | 0.991484 |
| Regression | Linear Regression | MAE | Approximately ₹46.33 |
| Regression | Linear Regression | RMSE | Approximately ₹57.27 |

Regression errors are stored in lakhs in reports/test_metrics.json.
The application converts them to rupees for display.

The direct classification rule has 100% agreement with its labels,
and the direct growth formula has zero error against its targets,
by construction. Linear Regression performs extremely well because
current price is an input and the regression target is its constant multiple.

## Application features

- Property detail form with dependent state and city choices.
- Classification and five-year projected-price outputs.
- Uncalibrated classification probability.
- Direct affordability benchmark and growth-formula comparison.
- Last submitted prediction retained during browsing.
- Filters for state, city, property type, price, size and BHK.
- Filtered CSV download.
- Price distribution and city comparison charts.
- Test performance and model coefficient charts.

## Project files

| File or folder | Purpose |
|---|---|
| app.py | Streamlit application |
| Real_Estate_Investment_Advisor.ipynb | Cleaning, EDA and modelling notebook |
| requirements.txt | Application environment packages |
| data/ | Cleaned property data and training-derived benchmarks |
| models/ | Saved preprocessing/model pipelines and metadata |
| reports/ | Evaluation results, split IDs and experiment summary |
| mlflow_tracking.db | MLflow experiment database |
| mlflow_artifacts/ | Model artifacts recorded in MLflow |

## Run locally on Windows

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically.

The application was verified locally using Python 3.11.9.
Training package versions are recorded in models/metadata.json.

## Reproduce the notebook

Upload the original india_housing_prices.csv to Google Colab,
then run the notebook cells in order.

The original CSV is required by the initial cleaning cells;
the cleaned CSV is not a substitute for that input.

## MLflow tracking

Nine experiment runs record validation comparisons and selected-model
test results. Existing results were logged after training.

Selected pipelines were stored as artifacts. This does not mean
they were registered in the MLflow Model Registry.

The application runs without MLflow installed. Working with the
tracking database separately requires MLflow; the recording session
used version 3.17.2.

The database contains artifact locations from the original Colab
environment. These paths may require updating before artifacts can
be opened through MLflow on another computer. A portable run summary
is available in reports/mlflow_runs.csv.

## Limitations

- Targets are constructed rather than observed investment outcomes.
- No sale-date history is available for historical price trends.
- Year_Built is construction year, not a price observation date.
- Crime rate, infrastructure score and RERA status are absent.
- Full-dataset EDA occurred before splitting, so evaluation is an
  educational holdout assessment rather than an external test.
- Classification probabilities have not been calibrated.
- Coefficient charts describe model weights, not causal effects.
- Projections exclude rental income, taxes, transaction fees and inflation.

## Technology

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib,
Streamlit and MLflow.