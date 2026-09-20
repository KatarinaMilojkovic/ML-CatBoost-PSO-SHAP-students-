# ========================================
#  Higher Education Students Performance – CatBoostRegressor model (ordinal formulation), Particle Swarm Optimization, and SHAP
#  Predictive and Interpretable Machine Learning Model for Modeling Students' Final Grade
# ========================================
"""
This research aims to share insights and knowledge regarding the development of a
predictive and interpretable machine learning model, CatBoostRegressor (loss: RMSE),
for predicting students' end-of-term final grades and
explaining the model’s predictions employing SHAP, an explainable AI method.
The problem is formulated as ORDINAL: GRADE (0: Fail ... 7: AA) is an ordered scale, so it is
modeled as a number 0-7 (regression with RMSE) instead of 8 unordered classes.
The continuous prediction is rounded (and clipped to 0-7) to obtain a grade.
As well as share the process of tuning CatBoostRegressor hyperparameters using
Particle Swarm Optimization (PSO), a population-based metaheuristic algorithm,
with 3-fold stratified cross-validation.

Dataset: The dataset used in this study is the "Higher Education Students
Performance Evaluation" dataset, collected from the Faculty of Engineering and
the Faculty of Educational Sciences students in 2019.
Source: Yilmaz, N. & Şekeroğlu, B. (2019). Higher Education Students Performance
Evaluation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51G82.
The dataset is a 145 x 33 table of categorical/integer data, where each column
corresponds to a question or attribute, and each row corresponds to a student.
Questions 1-10 are personal questions, questions 11-16 are family questions,
and questions 17-30 cover the student's education habits.
DATA.csv does not contain any helper columns that need to be dropped or broken down into multiple columns.
The target column is GRADE, the student's end-of-term output grade
(0: Fail, 1: DD, 2: DC, 3: CC, 4: CB, 5: BB, 6: BA, 7: AA). All remaining columns
(personal questions, family questions, education habit questions, and COURSE ID)
are used as categorical features, except STUDENT ID, which was dropped: it is
unique per row (145 distinct values for 145 students), so it carries no
generalizable signal and would only add noise/overfitting risk.

Symmetries & Asymmetries: Although the questionnaire exhibited structural symmetry in questionnaire design
through questions with a predefined set of answers, the response distribution for
question GRADE showed strong asymmetry, with the highest-engagement
category '1' being 4.375 times more than the lowest-engagement category '0'.
This was heavily imbalanced, which resulted in biased predictions.
The distribution of the GRADE target column is checked
below via a value count. Class imbalance across the eight grade categories is
expected, since some grades are naturally awarded far more often than others,
which can result in biased predictions if left unaddressed.

All stages of the study, including data loading, preprocessing,
model optimization, training, evaluation metrics, and interpretability,
were implemented in Python and executed in the Google Colab environment.
"""

# =====================================================================================
# SIR 2:
# ablation tabelu:
# default Random Forest;
# Random Forest + class weights;
# Random Forest + class weights + PSO;
# Random Forest + class weights + PSO + ensemble.
# Master:
# ablation tabelu:
# default Logistic Regression;
# Logistic Regression + class weights;
# Logistic Regression + class weights + PSO;
# Logistic Regression + class weights + PSO + ensemble.
# =====================================================================================



#--------------------------------------Installing & importing libraries
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, mean_absolute_error, cohen_kappa_score)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pyswarms as ps
from joblib import Parallel, delayed

print("\n\n============ PREPROCESSING... ============")
print("- The dataset is read from the CSV file and stored as a DataFrame object.")
print("- STUDENT ID column is dropped (unique per row, no generalizable signal).")
print("- The target is column 'GRADE', while all remaining columns are treated as categorical features.")
print("- ORDINAL formulation: GRADE (0: Fail ... 7: AA) is an ordered scale, so it is used as a number 0-7 (CatBoostRegressor, RMSE).")
DATA_PATH = "DATA.csv"
df = pd.read_csv(DATA_PATH)
df_clean = df.drop(columns=['STUDENT ID'])
target_col = 'GRADE'
#This will show True in any column where hidden whitespace exists:
#(df_clean.astype(str) != df_clean.astype(str).apply(lambda col: col.str.strip())).any()
#if everything is false:
X = df_clean.drop(columns=[target_col]).astype(str)  #features
y = df_clean[target_col].astype(str)                 #target (as text only for LabelEncoder / class weights / plot labels)
#if something is true:
#X = df_clean.drop(columns=[target_col]).astype(str).applymap(lambda v: v.strip())
#y = df_clean[target_col].astype(str).apply(lambda v: v.strip())
feature_names = list(X.columns); print("Features names (X):", feature_names)
cat_features = list(range(X.shape[1])); print("***Categorical features names:", cat_features)
le = LabelEncoder()
y_enc = le.fit_transform(y)              # 0..7 (same as GRADE); used only for class weights
class_labels = list(le.classes_)
y_ord = df_clean[target_col].to_numpy(dtype=float)   # ordinal numeric target 0..7 (regression target)
assert np.array_equal(y_enc, y_ord.astype(int)), "GRADE must equal its LabelEncoder code (0..7)"
N_GRADES = len(class_labels)
print("Target name (y):", target_col)
print("Grade values (ordinal 0..7):", class_labels)
print("--- Balanced:             class with highest value count <= 2-2,5 times class with lowest value count; classes have similar value counts")
print("--- Moderately balanced:  class with highest value count > 3-4 times class with lowest value count")
print("--- Severely imbalanced:  class with highest value count == 5-10 times class with lowest value count; class has < 20-30 value count")
print("Are grades balanced? Target value count: \n{}\n".format(df_clean[target_col].value_counts()))
print("- Conclusion: Severely imbalanced grades, '1' is 4.375 times more than '0'.")
print("- Balanced class weights are used as SAMPLE WEIGHTS (CatBoostRegressor has no class_weights): every student gets the weight of his/her grade.")
print("- Sample weights are incorporated into the CatBoost training process to ensure the model learns from all grades fairly.")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)
print("Class weights (per grade 0..7):", class_weights)

def sample_weights(y_arr):
    """Per-student weight = balanced class weight of his/her grade."""
    return class_weights[np.asarray(y_arr).astype(int)]

def to_grade(pred_cont):
    """Continuous regression output -> grade 0..7 (round + clip)."""
    return np.clip(np.rint(np.asarray(pred_cont).ravel()), 0, N_GRADES - 1).astype(int)









print("\n\n============ train_test_split... ============")
print("Why stratification of the target for train/test split?")
print("--- Split will be 80/20 but that does not guarantee that the target grade distribution will be the same in both sets.")
print("--- Stratification ensures that the target grade distribution is preserved in both training and test sets.")
print("--- In other words, it will not happen that the test set has all representations of one grade and the training set has 0.")
print("--- Stratification uses the ordinal grade values 0..7 as discrete labels (the target itself stays numeric for the regressor).")
print("--- --- The train_test_split makes a representative 80/20 train/test split.")
print("--- --- That representative train set is used by the fitness function to make representative 3 folds.")
print("--- --- That representative train set is used by the optimized model.")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_ord,
    test_size=0.2,
    stratify=y_ord,
    random_state=RANDOM_STATE
)
y_test_int = y_test.astype(int)
print("training set: {} students | test set: {} students".format(len(X_train), len(X_test)))





print("\n\n============ Particle Swarm Optimization (PSO)... ============")
print("For hyperparameter optimization, Particle Swarm Optimization (PSO) is used to identify an optimal learning rate, depth, L2 regularization, and random_strength.")
print("--- Imagine it like this: PSO optimizer randomly generates particles within defined bounds.")
print("--- Here, the pso optimizer is implemented using the Global Best topology ('global best pso optimizer').")
print("--- Global best influences particles movement -> particles are influenced to move toward the global-best particle (particle with the best fitnesse).")
print("--- The number of particles(10) and iterations(15) was chosen to balance optimization quality and computational efficiency.")
print("PSO optimizer proposes particles (hyperparameter combinations) and calls a custom fitness function to evaluate them.")
print("Here, the fitness function evaluates all particles in parallel using temporary CatBoostRegressor model (RMSE) with 3-fold stratified cross-validation on the training set only (80%).")
print("--- Why CatBoostRegressor?")
print("--- --- Because GRADE is an ordinal scale: predicting 6 when the truth is 7 is a smaller error than predicting 0. Regression (RMSE) respects that order, multiclass classification does not.")
print("--- Why k-fold stratified cross-validation strategy?")
print("--- --- To get a more reliable particle evaluation score (splitting the training set into 3 folds)")
print("--- --- Why spliting is better?")
print("--- --- --- Because splitting once, the score can depend on which students happen to be in the test set.")
print("--- --- --- When it is split 3 times, the score is more reliable.")
print("--- --- Why it has shuffle enabled?")
print("--- --- --- Because the folds could be affected by the original ordering of the data.")
print("--- --- --- The data will be shuffled before the folds are created")
print("--- --- StratifiedKFold creates folds while maintaining approximately the same grade distribution in each fold (like the strytify in train_test_split).")
print("--- --- Why 3-fold?")
print("--- --- --- The 3-fold setup provided a balance between computational efficiency and statistical reliability.")
print("--- Why only on training set?")
print("--- --- Because the test set is reserved for final evaluation and should not be used during hyperparameter optimization.")
print("--- --- The training set is split into 3 folds and the fitness function evaluation score is based on the train/test set in that split.")
print("The fitness function calculates the RMSE for each fold and returns the mean RMSE to PSO.")
print("The goal is to find the particle that gives the lowest evaluation score (mean RMSE across 3 folds).")
print("Why?")
print("--- RMSE is the typical size of the grade error (in grade points, big errors are punished more).")
print("--- The model needs to predict grades close to the real ones for all students.")
print("--- Students are weighted by their grade's balanced class weight during training, so rare grades are not ignored.")
print("Important thing to note: PSO minimizes the objective/fitness function.")
print("--- RMSE is already 'lower is better', so (unlike Macro F1 in the classifier version) it is NOT negated.")
print("A fixed random seed (42) was used to ensure reproducibility throughout hyperparameter evaluations.")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # smaller for speed

def pso_objective_parallel(particles):  #PSO parallelized objective function (objective==fitness)
    def evaluate_particle(p):
        lr = float(p[0])
        depth = int(round(p[1]))
        l2 = int(round(p[2]))
        random_strength = float(p[3])
        fold_scores = []

        for train_idx, test_idx in skf.split(X_train, y_train):# ===== Corection-2: only on training set =====
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]

            #pool - CatBoost’s optimized data structure used to store features, labels, categorical features and sample weights
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_features, weight=sample_weights(y_tr))
            te_pool = Pool(X_te, y_te, cat_features=cat_features, weight=sample_weights(y_te))

            model = CatBoostRegressor(
                iterations=500,  # smaller for speed
                learning_rate=lr,
                depth=depth,
                l2_leaf_reg=l2,
                random_strength=random_strength, #to resist overfitting on this small, noisy dataset
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=RANDOM_STATE,
                early_stopping_rounds=50,
                verbose=False,
                thread_count=-1,
                allow_writing_files=False
            )
            model.fit(tr_pool, eval_set=te_pool, use_best_model=True)
            preds = model.predict(te_pool)
            fold_scores.append(np.sqrt(mean_squared_error(y_te, preds)))

        return np.mean(fold_scores)   # RMSE: lower is better -> PSO minimizes it directly

    # Parallel evaluation of all particles
    return np.array(Parallel(n_jobs=-1)(delayed(evaluate_particle)(p) for p in particles))

"""PSO search space was defined in the code as parameter bounds
from 0.01 to 0.10 for learning rate,
from 3 to 10 for depth,
from 1 to 10 for L2 regularization,
from 0 to 5 for random_strength (added regularization dimension)
"""
bounds = (
    np.array([0.01, 3, 1, 0.0]),
    np.array([0.10, 10, 10, 5.0])
)
optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=4,
    options={'c1':1.4, 'c2':1.4, 'w':0.7},
    bounds=bounds
)
cost, pos = optimizer.optimize(pso_objective_parallel, iters=15)  # more thorough search
best_params = {
    'learning_rate': float(pos[0]),
    'depth': int(round(pos[1])),
    'l2_leaf_reg': int(round(pos[2])),
    'random_strength': float(pos[3])
}
print("\nBest parameters found by PSO after 15 iterations:")
print(best_params)
print("Best (lowest) mean 3-fold RMSE on the training set: {:.4f}".format(cost))








print("\n\n============ Oprimized CatBoostRegressor model... ============")
X_fit, X_val, y_fit, y_val = train_test_split( # ===== Corection-2: only on training set =====
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=RANDOM_STATE
)
print("(from training set) training set: {} students | (from training set) test set: {} students".format(len(X_fit), len(X_val)))

print("5 CatBoostRegressor models were independently trained with different random seeds and identical optimized hyperparameters found by PSO.")
print("--- This way we are not relying on one random training run.")
print("--- Imagine it like this: 5 teachers make a prediction for the same group of students. The predictions would be slightly different.")
models = []
for seed in [0, 1, 2, 3, 4]:
    m = CatBoostRegressor(
        iterations=2000,
        learning_rate=best_params['learning_rate'],
        depth=best_params['depth'],
        l2_leaf_reg=best_params['l2_leaf_reg'],
        random_strength=best_params['random_strength'],
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=seed,
        early_stopping_rounds=100,
        verbose=False,
        allow_writing_files=False
    )
    m.fit(
        Pool(X_fit, y_fit, cat_features=cat_features, weight=sample_weights(y_fit)),
        eval_set=Pool(X_val, y_val, cat_features=cat_features, weight=sample_weights(y_val)),
        use_best_model=True
    )
    models.append(m)

print("Every 5 model makes a prediction (a number on the 0-7 grade scale) for every test student from X_test.")
print("Because we now have 5 predictions for each test student, averaging was done (ensemble average = the regression counterpart of soft voting)")
pred_cont = np.mean( #ensemble average of the 5 continuous predictions
    [m.predict(Pool(X_test, y_test, cat_features=cat_features)) for m in models],
    axis=0
)
print("\nExample prediction for first test student: continuous {:.3f} -> grade {} (true grade {})".format(
    pred_cont[0], to_grade(pred_cont)[0], y_test_int[0]))
print("Now we have one continuous prediction for every test student but we want 'we predict this grade for this test student'.")
print("So the continuous prediction is rounded and clipped to the 0-7 grade scale (y_pred).")
y_pred = to_grade(pred_cont)








print("\n\n============ Evaluation metrics... ============")
print("y_test (real test students grades) vs pred_cont / y_pred (predicted continuous / rounded grades)")
def _metrics_from(y_true_arr, y_pred_cont):
    """Regression metrics on the continuous prediction + ordinal/classification metrics on the rounded grade."""
    yt = np.asarray(y_true_arr).astype(int)
    yc = np.asarray(y_pred_cont).ravel()
    yr = to_grade(yc)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yc))),
        "mae": float(mean_absolute_error(yt, yc)),
        "qwk": float(cohen_kappa_score(yt, yr, weights="quadratic", labels=list(range(N_GRADES)))),
        "accuracy": float(accuracy_score(yt, yr)),
        "within1": float(np.mean(np.abs(yt - yr) <= 1)),
        "macro_f1": float(f1_score(yt, yr, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(yt, yr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, yr, average="macro", zero_division=0)),
    }
METRIC_KEYS = ["rmse", "mae", "qwk", "accuracy", "within1", "macro_f1", "macro_precision", "macro_recall"]
LOWER_IS_BETTER = {"rmse", "mae"}   # all other metrics: higher is better

_m_opt = _metrics_from(y_test, pred_cont)
print(f"RMSE (continuous prediction, grade points): {_m_opt['rmse']:.4f}")
print(f"MAE  (continuous prediction, grade points): {_m_opt['mae']:.4f}")
print(f"QWK (quadratic weighted kappa, rounded)   : {_m_opt['qwk']:.4f}")
print(f"Accuracy (rounded grade)                  : {_m_opt['accuracy']:.4f}")
print(f"Within +-1 grade (rounded)                : {_m_opt['within1']:.4f}")
print(f"Macro F1 (rounded)                        : {_m_opt['macro_f1']:.4f}")
print(f"Macro-Precision (rounded)                 : {_m_opt['macro_precision']:.4f}")
print(f"Macro-Recall (rounded)                    : {_m_opt['macro_recall']:.4f}")

print("\nClassification Report (rounded predictions):")
print(classification_report(y_test_int, y_pred, labels=list(range(N_GRADES)), target_names=class_labels, zero_division=0))

print("\nConfusion matrix (rounded predictions):")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
print(confusion_matrix(y_test_int, y_pred, labels=list(range(N_GRADES))))
cm = confusion_matrix(y_test_int, y_pred, labels=list(range(N_GRADES)))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
print("--- Conclusion: for an ordinal target look at how far the predictions are from the diagonal (MAE, RMSE, within +-1, QWK), not only at exact hits.")



print("How reliable are the metrics we got on these 29 test students?")
print("Test set of 29 instances is too small for strong conclusions.")
print("That's why we use Bootstrap 95% confidence interval (2.5 / 50 / 97.5 percentil)")
print("So, before this point, we trained the model, we gave it the test set (29 instances) and got predictions.")
print("Now, we give a bootstrap test set (29 instances but some are repeated). ")
print("We call the model 2000 times with a different bootstrap test set (29 instances but some are repeated)")
print("And show the bootstrap prediction [50] and the 95% interval [2.5, 97.5]")
print("The result could vary if you had a slightly different sample of test students -> Bootstrap gives you an estimate of that uncertainty")
BOOT_N = 2000
_boot_rng = np.random.default_rng(RANDOM_STATE)

def bootstrap_ci(y_true_arr, y_pred_cont, n=BOOT_N):
    y_true_arr = np.asarray(y_true_arr); y_pred_cont = np.asarray(y_pred_cont).ravel()
    acc = {k: [] for k in METRIC_KEYS}
    m = len(y_true_arr)
    for _ in range(n):
        idx = _boot_rng.integers(0, m, m)
        if len(np.unique(y_true_arr[idx])) < 2:   # degenerate resample -> skip
            continue
        r = _metrics_from(y_true_arr[idx], y_pred_cont[idx])
        for k in METRIC_KEYS:
            if np.isfinite(r[k]):   # e.g. QWK can be undefined (0/0) on a degenerate resample
                acc[k].append(r[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 50), np.percentile(v, 97.5)) for k, v in acc.items()}

def print_ci(title, ci):
    print("\n--- {} ---".format(title))
    for k, (lo, med, hi) in ci.items():
        print("  {:<16}: {:.4f}   95% CI [{:.4f}, {:.4f}]".format(k, med, lo, hi))

# saving optimized predictions
pred_cont_opt = np.asarray(pred_cont).ravel().copy()

ci_opt = bootstrap_ci(y_test, pred_cont_opt)
print_ci("Bootstrap 95% CI Optimized model (ensemble average)", ci_opt) # ===== Corection-1: Bootstrap 95% CI Optimized model =====

















print("\n\n============ SHAP...CatBoost’s built-in feature importance... ============")
print("Interpreting the predictions of 5 models")
print("Regression output: one SHAP value per feature and student (no class dimension), in grade points.")
print("--- SHAP importance")
print("Because it is model-specific, SHAP importance (mean SHAP and mean |SHAP|) was calculated for each model")
print("The mean |SHAP| values were calculated across all students to obtain global feature importance scores.")
print("After that, it calculates the mean values of those 5 models")
print("--- CatBoost’s built-in feature importance")
print("Because it is model-specific, CatBoost’s built-in feature importance was calculated for each model")
print("After that, it calculates the mean values of those 5 models")

train_pool = Pool(X_train, y_train, cat_features=cat_features)
n_features = len(feature_names)

_per_model_sv = []          # (n_samples, n_features) SHAP arrays per model
_per_model_mean_shap = []
_per_model_mean_abs_shap = []
_per_model_cb = []
for _m in models:
    _sv = np.asarray(_m.get_feature_importance(train_pool, type='ShapValues'))[:, :-1]  # remove the baseline (expected value) column
    _per_model_sv.append(_sv)
    _per_model_mean_shap.append(np.mean(_sv, axis=0))
    _per_model_mean_abs_shap.append(np.mean(np.abs(_sv), axis=0))
    _per_model_cb.append(np.array(_m.get_feature_importance(train_pool, type='FeatureImportance')))

# usrednjeno preko 5 modela = objasnjenje finalnog (ensemble) prediktora
mean_shap = np.mean(_per_model_mean_shap, axis=0)
mean_abs_shap = np.mean(_per_model_mean_abs_shap, axis=0)
cb_importance = np.mean(_per_model_cb, axis=0)
# standardna devijacija preko 5 modela = mera stabilnosti vaznosti unutar ansambla
mean_abs_shap_std = np.std(_per_model_mean_abs_shap, axis=0)
cb_importance_std = np.std(_per_model_cb, axis=0)
# usrednjeni po-uzorak SHAP niz za dependence/histogram grafike nize
shap_vals_no_base = np.mean(_per_model_sv, axis=0)   # (n_samples, n_features)

_stab = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap_across_5': mean_abs_shap,
    'std_of_mean_abs_shap_across_5': mean_abs_shap_std,
    'cb_importance_across_5': cb_importance,
    'std_of_cb_importance_across_5': cb_importance_std,
}).sort_values('mean_abs_shap_across_5', ascending=False)
print("\nThe feature importance table (mean and std):")
print(_stab.head(12).to_string(index=False)) # ===== Corection-4: interpretaion of the final prediction (mean and std) =====







#visual helpers
print(f"\nAll features influencing {target_col} - sort(mean_abs_shap):")
fi_df_mas = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap,
    'mean_abs_shap': mean_abs_shap,
    'cb_importance': cb_importance
}).sort_values('mean_abs_shap', ascending=False)
print(fi_df_mas)
#mean_abs_shap
plt.figure(figsize=(8,10))
plt.barh(fi_df_mas.feature, fi_df_mas.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.barh(fi_df_mas.feature, fi_df_mas.mean_shap, alpha=0.6, label='Mean SHAP')
plt.barh(fi_df_mas.feature, fi_df_mas.cb_importance, alpha=0.4, label='CatBoost Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.show()
print(f"\nAll features influencing {target_col} - sort(mean_shap):")
fi_df_ms = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap,
    'mean_abs_shap': mean_abs_shap,
    'cb_importance': cb_importance
}).sort_values('mean_shap', ascending=False)
print(fi_df_ms)
#mean_shap
plt.figure(figsize=(8,10))
plt.barh(fi_df_ms.feature, fi_df_ms.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.barh(fi_df_ms.feature, fi_df_ms.mean_shap, alpha=0.6, label='Mean SHAP')
plt.barh(fi_df_ms.feature, fi_df_ms.cb_importance, alpha=0.4, label='CatBoost Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.show()
print(f"\nAll features influencing {target_col} - sort(cb_importance):")
fi_df_cb = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap,
    'mean_abs_shap': mean_abs_shap,
    'cb_importance': cb_importance
}).sort_values('cb_importance', ascending=False)
print(fi_df_cb)
#cb_importance
plt.figure(figsize=(8,10))
plt.barh(fi_df_cb.feature, fi_df_cb.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.barh(fi_df_cb.feature, fi_df_cb.mean_shap, alpha=0.6, label='Mean SHAP')
plt.barh(fi_df_cb.feature, fi_df_cb.cb_importance, alpha=0.4, label='CatBoost Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.show()
#-----separate----
plt.figure(figsize=(8,10))
plt.barh(fi_df_mas.feature, fi_df_mas.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,10))
plt.barh(fi_df_ms.feature, fi_df_ms.mean_shap, alpha=0.6, label='Mean SHAP')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,10))
plt.barh(fi_df_cb.feature, fi_df_cb.cb_importance, alpha=0.4, label='CatBoost Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.show()







#To understand how weekly study hours (column "17") influence predictions:
import shap
shap_grade = shap_vals_no_base  # (n_samples, n_features): SHAP values in grade points (no class dimension in regression)
shap.dependence_plot(
    "17",
    shap_grade,
    X_train,
    feature_names=feature_names
)
#or check distribution:
study_hours_idx = feature_names.index("17")
plt.hist(shap_grade[:, study_hours_idx], bins=50)
plt.title("SHAP value distribution for weekly study hours - column 17 (grade points)")
plt.xlabel("SHAP value (grade points)")
plt.ylabel("Frequency")
plt.show()










print("\n=== Default CatBoostRegressor Baseline (baseline for PSO and optimized model evaluation) ===")
print("- differences between baseline and optimized model are PSO tuning, sample (class) weights, and the 5-seed ensemble")
baseline_model = CatBoostRegressor(
    random_state=42,
    verbose=0,
    allow_writing_files=False
)
baseline_model.fit(Pool(X_train, y_train, cat_features=cat_features)) # Train
baseline_test_pool = Pool(X_test, y_test, cat_features=cat_features)
baseline_cont = np.asarray(baseline_model.predict(baseline_test_pool)).ravel() # Predict (continuous)
y_pred = to_grade(baseline_cont)

print("\n=== Evaluation Metrics ===")
_m_base = _metrics_from(y_test, baseline_cont)
print(f"RMSE (continuous prediction, grade points): {_m_base['rmse']:.4f}")
print(f"MAE  (continuous prediction, grade points): {_m_base['mae']:.4f}")
print(f"QWK (quadratic weighted kappa, rounded)   : {_m_base['qwk']:.4f}")
print(f"Accuracy (rounded grade)                  : {_m_base['accuracy']:.4f}")
print(f"Within +-1 grade (rounded)                : {_m_base['within1']:.4f}")
print(f"Macro-F1 (rounded)                        : {_m_base['macro_f1']:.4f}")
print(f"Macro-Precision (rounded)                 : {_m_base['macro_precision']:.4f}")
print(f"Macro-Recall (rounded)                    : {_m_base['macro_recall']:.4f}")

print("\n=== Classification Report (rounded predictions) ===")
print(classification_report(y_test_int, y_pred, labels=list(range(N_GRADES)), target_names=class_labels, zero_division=0))

print("\n=== Confusion Matrix (rounded predictions): ===")
print(confusion_matrix(y_test_int, y_pred, labels=list(range(N_GRADES))))
cm = confusion_matrix(y_test_int, y_pred, labels=list(range(N_GRADES)))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


ci_base = bootstrap_ci(y_test, baseline_cont)
print_ci("Bootstrap 95% CI Default model", ci_base) # ===== Corection-1: bootstrap CI for default model (!!! compare optimized vs base bootstrap)=====
print("\n--- Comparison of Optimized and Default Models Using Bootstrap CI on the same resampled test set ---")
_yt = np.asarray(y_test)
_pair_rng = np.random.default_rng(RANDOM_STATE)
_diffs = {k: [] for k in METRIC_KEYS}
_m = len(_yt)
for _ in range(BOOT_N):
    _idx = _pair_rng.integers(0, _m, _m)
    if len(np.unique(_yt[_idx])) < 2:
        continue
    _ro = _metrics_from(_yt[_idx], pred_cont_opt[_idx])
    _rb = _metrics_from(_yt[_idx], baseline_cont[_idx])
    for _k in METRIC_KEYS:
        if np.isfinite(_ro[_k]) and np.isfinite(_rb[_k]):
            _diffs[_k].append(_ro[_k] - _rb[_k])
for _k in METRIC_KEYS:
    _d = np.array(_diffs[_k])
    # For RMSE and MAE, lower values indicate better performance.
    # For all other metrics, higher values indicate better performance.
    _better = np.mean(_d < 0) if _k in LOWER_IS_BETTER else np.mean(_d > 0)
    print("  {:<16}: median difference {:+.4f}   95% CI [{:+.4f}, {:+.4f}]   "
        "proportion of resamples favoring the optimized model: {:.1%}".format(
        _k,
        np.percentile(_d, 50),
        np.percentile(_d, 2.5),
          np.percentile(_d, 97.5),
          _better))











print("\n\n============ Incremental ablation study (adding one component at a time) ============")
print("The evaluation is on the same y_test set")
# ===== Corection-3: ablation table =====
#   A) default CatBoostRegressor
#   B) A + sample (class) weights
#   C) B + PSO hyperparameters (testing a single CatBoostRegressor model using the hyperparameters already found by PSO)
#   D) C + ensemble 5 modela with averaging (= optimized ensemble)
def _eval_block(name, y_true_arr, y_pred_cont):
    r = _metrics_from(np.asarray(y_true_arr), np.asarray(y_pred_cont).ravel())
    print("  {:<62} rmse={:.4f}  mae={:.4f}  qwk={:.4f}  acc={:.4f}  within1={:.4f}  macroF1={:.4f}  macroP={:.4f}  macroR={:.4f}".format(
        name, r["rmse"], r["mae"], r["qwk"], r["accuracy"], r["within1"], r["macro_f1"], r["macro_precision"], r["macro_recall"]))
    return r

# A)
_A = CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False)
_A.fit(Pool(X_train, y_train, cat_features=cat_features))
_eval_block("A) default CatBoostRegressor", y_test, _A.predict(baseline_test_pool))

# B)
_B = CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False)
_B.fit(Pool(X_train, y_train, cat_features=cat_features, weight=sample_weights(y_train)))
_eval_block("B) default CatBoostRegressor + sample (class) weights", y_test, _B.predict(baseline_test_pool))

# C)
_C = CatBoostRegressor(
    iterations=2000,
    learning_rate=best_params['learning_rate'],
    depth=best_params['depth'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    random_strength=best_params['random_strength'],
    loss_function='RMSE', eval_metric='RMSE',
    random_seed=0, early_stopping_rounds=100, verbose=False,
    allow_writing_files=False
)
_C.fit(Pool(X_fit, y_fit, cat_features=cat_features, weight=sample_weights(y_fit)),
       eval_set=Pool(X_val, y_val, cat_features=cat_features, weight=sample_weights(y_val)), use_best_model=True)
_eval_block("C) default CatBoostRegressor + sample weights + PSO hyperparameters", y_test, _C.predict(baseline_test_pool))

# D)
_eval_block("D) default CatBoostRegressor + sample weights + PSO hyperparameters + ensemble 5 modela (average)", y_test, pred_cont_opt)









# #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Data verification from AI project
# #DataFrame object
# print(type(df))
# print("Loaded:", df.shape)
# print("Columns:", list(df.columns))
# #uzorak data seta:
# print("head: \n{}\n".format(df_clean.head())) #print(df_clean[0:5]) #ispisi prvih 5 redova iz dataframe-a
# print("tail: \n{}\n".format(df_clean.tail)) #ispisivanje poslednjih 5 redova
# #indeksi pocinju od 0, korak je 1,
# print("index: {}\n".format(df_clean.index))
# print("1. red dataframe-a: \n{}\n".format(df_clean.iloc[0])) #1. red dataframe-a
# #dimenzija dataframe-a (2D):
# print("DataFrame ima %d dimenzije" % df_clean.ndim)
# #ispisi featur-e/kolone, target-e/labele, vrednosti
# print("Featurs/Kolone: ", df_clean.columns)
# print(f"Odgovori za {target_col} (target/label variable): ", df_clean[target_col].unique())
# print("Vrednosti: \n", df_clean.values) #prosledjuje samo vrednosti
# #koliko ima redova i kolona
# (row, col) = df_clean.shape
# print("Dataframe ima %d zapisa/redova i %d obelezja/kolona(labels)\n" % (row, col))
# print("Dataframe ima %d zapisa/redova i %d obelezja/kolona(labels)\n" % (len(df_clean), len(df_clean.columns)))
# #transponovan dataframe
# print("Transponovan dataframe: \n{}".format(df_clean.T))
# #sve max i min vrednosti kolona
# import numpy as np
# max = df_clean.apply(np.max)
# min = df_clean.apply(np.min)
# print("max vrednosti kolona: \n{}\n".format(max))
# print("min vrednosti kolona: \n{}\n".format(min))
# print(df_clean.info())
# print("(.info) --- Zakljucak:\nDataFrame ima %d redova i %d kolona\nVidimo kako su imenovane kolone\nKolone imaju int podatke (int64)\nsvaka kolona ima non-null vrednosti, tj. nemamo nedostajuce vrednosti\n" % (row, col))
# print(df_clean.describe()) #osnovne statisticke podatke
# print("(.describe) --- Zakljucak:\ncount: broj non-null podataka za svaku kolonu, nemamo prazna polja\nmean(avg): srednja vrednost svakog pitanja\nstd(standardna devijacija): disperzija vrednosti svake kolone oko mean, veci std ukazuje na vecu disperziju oko mean vrednosti\nmin,max: najmanja i najveca vrednost za svaku kolonu\n%: distribucija podataka, npr 25% ispitanika je izabralo odredjenu vrednost za odredjeno pitanje\n")
# print(df_clean.corr())
# print(f"(.corr) --- Zakljucak:\nprocena stepena linearnog odnosa izmedju 2 numbericke variable\nblizu 1: jaka pozitivna linearna veza, v1 se povecava kako se v2 povecava\nblizu-1: jaka negativna linearna veza, v1 se povecava kako se v2 povecava\nblizu 0: slaba ili nikakva linearna veza\npogledati korelacije kolona sa {target_col}\n")
