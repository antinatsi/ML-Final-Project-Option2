# ML-Final-Project-Option2
This portfolio is a short, organised presentation of the machine learning course final project. 
# Final Project: Does a Neural Network Beat a Classical Model?
**Course:** Machine Learning with Python (LEAP Erasmus+)  
**Track:** Option 2 – Tabular Classification on Adult Census Income  

---

## 1. Project Title & Description
This project evaluates whether a deep dense neural network provides sufficient performance gains over a traditional, interpretable classical model (Logistic Regression) on a real-world tabular dataset with mixed numeric and categorical features.

---

## 2. Problem Statement
The scenario involves a public agency seeking to identify households likely to fall within a higher-income bracket (> $50,000/year) using survey data to target assistance or outreach programmes. 

The central question investigated is:
> *Does a neural network outperform a well-regularised classical logistic regression model, and is the added complexity justified in a public decision-making context?*

---

## 3. Dataset Overview
- **Dataset:** Adult (Census Income) from OpenML (`adult`, version 2).
- **Target:** Binary classification of income (`<=50K` as `0`, `>50K` as `1`).
- **Class Balance:** Imbalanced (~76% in class `0`, ~24% in class `1`).
- **Features:**
  - **Numeric Features:** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
  - **Categorical Features:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.

---

## 4. Methodology & Workflow
1. **Train/Test Splitting:** Stratified 80/20 train-test split (`test_size=0.2`, `random_state=42`) to maintain class proportions.
2. **Preprocessing Pipeline (`ColumnTransformer`):**
   - *Numeric pipeline:* `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - *Categorical pipeline:* `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Classical Model:**
   - Preprocessor combined with `LogisticRegression(max_iter=1000)` inside a scikit-learn `Pipeline`.
   - Evaluated using 5-Fold Stratified Cross-Validation on the training set, scoring primarily by **F1-score**.
4. **Neural Network Model:**
   - Architecture: Input Layer (dimension equal to one-hot encoded features) $\rightarrow$ Dense (64 units, ReLU) $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (32 units, ReLU) $\rightarrow$ Output Dense (1 unit, Sigmoid).
   - Training: Compiled with `adam` optimizer and `binary_crossentropy` loss; trained for up to 20 epochs (`batch_size=128`, `validation_split=0.1`) with `EarlyStopping(patience=3, restore_best_weights=True)`.

---

## 5. Key Results

| Metric | Logistic Regression | Neural Network |
| :--- | :---: | :---: |
| **5-Fold CV F1 Score (Train)** | ~0.655 ± 0.007 | — |
| **Test F1 Score** | **~0.655** | **~0.675** |
| **Test Accuracy** | **~85.2%** | **~85.8%** |

- **Metric Focus:** Because the target is imbalanced (~76% majority class), **F1-score** was used as the primary metric. Accuracy alone is misleading because a trivial baseline predicting only class `0` would achieve ~76% accuracy with an F1 score of `0.0`.
- **Confusion Matrix Analysis:** Both models showed comparable false positive rates, but the Neural Network reduced false negatives by capturing more true positive cases (`>50K`).

---

## 6. Interpretation & Ethical Considerations
- **Performance:** The neural network yielded a modest ~2% gain in F1 score by automatically learning non-linear feature interactions between variables like age, education, and weekly working hours.
- **Fairness & Historical Bias:** The dataset incorporates demographic indicators (race, sex, country of origin). Models trained on such data risk codifying and propagating historical wage disparities.
- **Cost of Errors:** In public sector allocations, a false negative denies aid to an eligible applicant, while a false positive misallocates public funds.
- **Interpretability vs. Complexity:** In sensitive public applications, the transparency of logistic regression (where feature coefficients can be audited and explained) often outweighs the marginal statistical advantage of a black-box neural network.

---

## 7. Reflection & Future Improvements
- **What worked well:** End-to-end preprocessing pipelines prevented data leakage between splits, and `EarlyStopping` effectively prevented the neural network from overfitting.
- **Challenges:** Managing high-cardinality categorical variables via one-hot encoding expanded feature dimensionality significantly.
- **Potential improvements:**
  1. Experimenting with gradient-boosted decision trees (e.g., LightGBM or XGBoost), which typically dominate tabular benchmarks.
  2. Implementing threshold-tuning based on asymmetric costs of false positives vs. false negatives.
  3. Conducting demographic parity and equalized odds fairness audits across protected subgroups.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
   cd <repo-name>
