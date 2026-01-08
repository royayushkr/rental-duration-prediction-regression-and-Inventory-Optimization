# DVD Rental Duration Prediction

## Project Overview
A DVD rental company requires a predictive model to estimate the number of days a customer will rent a movie. Accurate predictions are essential for efficient inventory planning. The objective of this project is to build a regression model that predicts rental duration with a Mean Squared Error (MSE) of **3 or less** on the test set.

## Dataset
The project utilizes `rental_info.csv`, which contains transaction and movie data.

**Key Features:**
* **Dates:** `rental_date`, `return_date`.
* **Financial:** `amount`, `rental_rate`, `replacement_cost`.
* **Movie Details:** `release_year`, `length`, `special_features` (e.g., deleted scenes, behind the scenes).
* **Ratings:** Dummy variables for MPAA ratings (`NC-17`, `PG`, `PG-13`, `R`).

## Methodology

### 1. Data Preprocessing & Feature Engineering
* **Duration Calculation:** Created the target variable `rental_length_days` by subtracting `rental_date` from `return_date`.
* **Text Parsing:** Extracted binary features from the `special_features` column to create `deleted_scenes` and `behind_the_scenes` dummy variables.
* **Leakage Prevention:** Dropped columns that directly imply the target or are not available at the time of prediction (e.g., `return_date`, `rental_length`).

### 2. Feature Selection (Lasso Regression)
To identify the most impactful features, a Lasso regression model (`alpha=0.3`) was trained. Only features with positive coefficients were selected for the subsequent modeling phase to reduce noise and dimensionality.

### 3. Model Selection & Tuning
Two primary models were evaluated on the selected features:
1.  **Linear Regression (OLS):** Used as a baseline using the Lasso-selected features.
2.  **Random Forest Regressor:** Tuned using `RandomizedSearchCV` with 5-fold cross-validation.
    * **Hyperparameter Space:**
        * `n_estimators`: 1 to 100
        * `max_depth`: 1 to 10.


## Results
The **Random Forest Regressor** outperformed the linear baseline and met the client's success criteria.

| Model | MSE | Parameters |
| :--- | :--- | :--- |
| **Random Forest** | **2.226** | `n_estimators=51`, `max_depth=10` |

*The final model achieved an MSE of ~2.23, successfully surpassing the target threshold of 3.*

## Technologies Used
* **Python**: Core programming language.
* **pandas**: Data manipulation and time-series calculation (`dt.days`).
* **scikit-learn**:
    * **Models:** `Lasso`, `LinearRegression`, `RandomForestRegressor`.
    * **Tuning:** `RandomizedSearchCV`.
    * **Metrics:** `mean_squared_error`.

## Usage
1.  Ensure `rental_info.csv` is in the project directory.
2.  Run the Jupyter Notebook to execute the pipeline.
3.  The script will output the best model parameters and the final Mean Squared Error.
