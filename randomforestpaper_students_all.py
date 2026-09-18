# -*- coding: utf-8 -*-
"""RandomForestPaper_Students_All.py

Adapted from CatBoostPaper_FinalNoMildOversampling.ipynb so the two scripts
can be compared directly: same dataset, same preprocessing philosophy, same
PSO -> optimized-ensemble -> evaluation -> bootstrap CI -> ablation -> SHAP
pipeline, with CatBoostClassifier replaced by scikit-learn's
RandomForestClassifier.
"""

# ========================================
#  Higher Education Students Performance – Random Forest model, Particle Swarm Optimization, and SHAP
#  Predictive and Interpretable Machine Learning Model for Modeling Students' Final Grade
# ========================================
"""
This script mirrors catboostpaper_students_all_repeated_stratified_k_fold.py
so that a Random Forest model can be compared directly against the
CatBoostClassifier results on the same data, same splits, same random seeds,
and the same evaluation/ablation/SHAP methodology.

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

Unlike CatBoost, RandomForestClassifier cannot consume raw categorical columns
directly - every feature must be numeric. To keep the comparison fair (no
information CatBoost had access to is hidden from Random Forest, and no false
ordinal relationship is invented for what are really categorical answer
codes), every feature is one-hot encoded inside a scikit-learn Pipeline. The
encoder is fit ONLY on the training fold/split it is used with (never on data
it will later be evaluated on), exactly mirroring how CatBoost's Pool/
cat_features handling never leaks test information into training.

All stages of the study, including data loading, preprocessing,
model optimization, training, evaluation metrics, and interpretability,
were implemented in Python.
"""



#-----------------------------------------------------Installing & importing libraries
"""
This cell installs dependencies when run in Google Colab (the `!pip install` line
below is a Colab/IPython shell escape, not valid standalone Python, and Colab starts
each session with a fresh environment so it needs reinstalling every time). When
running this script locally instead, install the same dependencies once from a
terminal and skip this line:
    pip install pyswarms joblib scikit-learn pandas numpy matplotlib seaborn shap
"""
try:
    import google.colab  # only importable inside Google Colab
    IN_COLAB = True
    get_ipython().system('pip install --quiet pyswarms joblib')
except ImportError:
    IN_COLAB = False  # running locally (plain .py script or local Jupyter)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pyswarms as ps
from joblib import Parallel, delayed
import shap
import os

# Every plt.show() below is preceded by a plt.savefig() into this folder, so the
# figures used in the paper are reproducible from a plain script run (no need to
# manually screenshot the interactive windows). plt.show() itself is untouched, so
# interactive use (a GUI backend) still pops up the windows exactly as before.
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "(1-30)RandomForest", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)



#-----------------------------------------------------
# PREPROCESSING
#-----------------------------------------------------
print("\n\n============ PREPROCESSING... ============")

"""
The dataset was read from the CSV file and stored as a DataFrame object (df).
In Colab, DATA.csv is read from Google Drive (mounted first). Running locally,
DATA.csv is read from the working directory instead - place it next to this
script, or edit DATA_PATH below to point at wherever it lives on disk.
"""
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/gdrive")
    DATA_PATH = "/content/gdrive/MyDrive/DATA.csv"
else:
    DATA_PATH = "DATA.csv"
df = pd.read_csv(DATA_PATH)
"""
STUDENT ID was dropped: it is unique per row (one value per student), so it carries
no generalizable signal and would only add noise/overfitting risk to the model.
"""
df_clean = df.drop(columns=['STUDENT ID'])
print("STUDENT ID column dropped (unique per row, no generalizable signal)")
"""The categorical target was defined as GRADE, while all remaining columns as categorical features."""
target_col = 'GRADE'
print("Target column:", target_col)
X = df_clean.drop(columns=[target_col]).astype(str)  #features
y = df_clean[target_col].astype(str)                 #target
"""
The categorical target values were encoded using LabelEncoder to transform class labels into
integer labels suitable for multi-class classification.
The original class labels were preserved for prediction inter-pretation and evaluation metrics.
"""
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_labels = list(le.classes_)
print("Encoded classes (model order):", class_labels)
feature_names = list(X.columns)
print("Feature names:", feature_names)
cat_features = list(range(X.shape[1]))
print("Categorical features:", cat_features)
"""
A target value count was performed, and the presence of a large class imbalance was determined.
In order to mitigate the effects of target class imbalance, balanced class weights were computed
(and passed to RandomForestClassifier's class_weight= parameter as a {class: weight} dict, the
scikit-learn equivalent of CatBoost's class_weights= list, so the model learns from all classes fairly.)
"""
print("Balansirano: najveca klasa <= 2-2,5 puta najmanje klase; klase imaju slican broj primera")
print("Umereno balansirano: najveca klasa > 3-4 puta najmanje klase")
print("Veoma nebalansirano: najveca klasa === 5-10 puta najmanje klase; klasa ima < 20-30 primera")
print("balanced classes?: \n{}\n".format(df_clean[target_col].value_counts()))
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)
class_weight_dict = dict(zip(np.unique(y_enc), class_weights))
print("Class weights:", class_weights)
print("Class weight dict (for sklearn class_weight=):", class_weight_dict)



#-----------------------------------------------------
# One-hot encoding helper (RandomForestClassifier needs numeric input)
#-----------------------------------------------------
"""
Every model below is wrapped as a scikit-learn Pipeline: a ColumnTransformer
that one-hot encodes all (categorical) feature columns, feeding a
RandomForestClassifier. handle_unknown='ignore' means a category seen only at
prediction time (never seen in that particular training fold) is encoded as
all-zeros instead of raising an error. Building a *new* pipeline per fit
(instead of reusing one fitted encoder) guarantees the encoder is always fit
only on the data it is about to train on - never on validation or test rows -
matching CatBoost's leak-free Pool handling.
"""
def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:  # older scikit-learn used `sparse=` instead of `sparse_output=`
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def make_rf_pipeline(n_estimators=100, max_depth=None, min_samples_split=2,
                      min_samples_leaf=1, max_features='sqrt', class_weight=None,
                      random_state=None):
    """Build a fresh OneHotEncoder + RandomForestClassifier pipeline."""
    return Pipeline([
        ('prep', ColumnTransformer([('cat', _make_ohe(), cat_features)])),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        ))
    ])




#-----------------------------------------------------
# METAHEURISTIC OPTIMIZATION
# Particle Swarm Optimization (PSO) was employed to identify optimal max_depth,
# min_samples_split, min_samples_leaf, and max_features for RandomForestClassifier.
#-----------------------------------------------------
print("\n\n============ Particle Swarm Optimization (PSO)... ============")

"""
The custom parallelized fitness function evaluates hyperparameters by
training temporary RandomForestClassifier models using stratified
cross-validation.

To ensure reproducibility, a fixed random seed was used throughout all hyperparameter evaluations.

Stratified K-fold cross-validation with three folds, repeated three times (9 fold
evaluations in total), was applied to enforce symmetry in the target class distribution
across folds - identical scheme to the CatBoost script, for a like-for-like comparison.

The train/test split is carved out BEFORE PSO, and the repeated 3-fold CV inside the
fitness function runs ONLY over the training set (X_train / y_train). The held-out test
set is never seen during hyperparameter search.
"""
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
)
print("\nTrening skup: {} studenata | Test skup: {} studenata "
      "(test skup se NE koristi ni u PSO-u ni za rano zaustavljanje)".format(len(X_train), len(X_test)))

kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=RANDOM_STATE)  # 9 fold evals for a more stable fitness signal
def pso_objective_parallel(particles):
    def evaluate_particle(p):
        max_depth = int(round(p[0]))
        min_samples_split = int(round(p[1]))
        min_samples_leaf = int(round(p[2]))
        max_features = float(np.clip(p[3], 0.05, 1.0))
        fold_scores = []

        for train_idx, test_idx in kf.split(X_train, y_train):
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]

            """
            Temporary RandomForestClassifier models were configured with
            PSO-tuned max_depth, min_samples_split, min_samples_leaf, and
            max_features, 500 trees (smaller for speed, mirroring CatBoost's
            500-iteration PSO fitness models), fixed random_state, and
            previously calculated balanced class_weight.
            """
            pipe = make_rf_pipeline(
                n_estimators=500,  # smaller for speed
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight_dict,
                random_state=RANDOM_STATE
            )
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            """
            The performance was measured using the mean macro F1-score,
            suitable for class imbalance because it treats all classes equally.
            """
            fold_scores.append(f1_score(y_te, preds, average='macro'))

        """Minimizing the mean macro F1-score across folds resulted in max-imizing Macro F1."""
        return -np.mean(fold_scores)

    # Parallel evaluation of all particles
    return np.array(Parallel(n_jobs=-1)(delayed(evaluate_particle)(p) for p in particles))

"""PSO search space was defined in the code as parameter bounds
from 3 to 20 for max_depth,
from 2 to 20 for min_samples_split,
from 1 to 10 for min_samples_leaf,
from 0.05 to 1.0 for max_features (fraction of one-hot features considered per split)
"""
bounds = (
    np.array([3, 2, 1, 0.05]),
    np.array([20, 20, 10, 1.0])
)
"""
*The optimizer was implemented using the Global Best topology
with 10 particles, 4 dimensions, defined parameter bounds, and set options.
*The custom parallelized fitness function^^^ simultaneously evaluated
all hyperparameter sets proposed by the optimizer and returned their performance scores.
The optimizer relied on those performance scores for returning the
optimal-performing hyperparameter combination after 15 iterations.
"""
optimizer = ps.single.GlobalBestPSO(
    n_particles=10,
    dimensions=4,
    options={'c1':1.4, 'c2':1.4, 'w':0.7},
    bounds=bounds
)
cost, pos = optimizer.optimize(pso_objective_parallel, iters=15)  # more thorough search
best_params = {
    'max_depth': int(round(pos[0])),
    'min_samples_split': int(round(pos[1])),
    'min_samples_leaf': int(round(pos[2])),
    'max_features': float(np.clip(pos[3], 0.05, 1.0))
}
print("\nBest parameters found by PSO after 15 iterations:")
print(best_params)




#-----------------------------------------------------
# Optimized RandomForestClassifier model
#-----------------------------------------------------
print("\n\n============ Optimized RandomForestClassifier model... ============")

"""
In this study, five RandomForestClassifier models were independently trained
with a different random seed and identical optimized hyperparameters, which
ensures performance differences came only from randomness (random_state
controls both bootstrap resampling of rows and the random feature subset
considered at each split).

Unlike CatBoost's gradient-boosted trees, a Random Forest does not overfit as
more trees are added - each tree is grown independently on a bootstrap sample
and averaging more of them only reduces variance, it never needs "early
stopping on a validation set" the way boosting does. So there is no internal
X_fit/X_val split here: each ensemble member is fit directly on the full
training set (X_train / y_train), maximizing the data available to each
independently-seeded model. The final forest size (1000 trees) is larger than
the 500 used inside the PSO fitness loop (which is kept smaller purely for
search speed) since more trees only help, never hurt, at prediction time.
"""
models = []
for seed in [0, 1, 2, 3, 4]:
    m = make_rf_pipeline(
        n_estimators=1000,
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        class_weight=class_weight_dict,
        random_state=seed
    )
    m.fit(X_train, y_train)
    models.append(m)

"""
Final predictions were obtained with a soft voting approach by averaging
class probabilities across models, which reduced prediction variance and
improved robustness, particularly for minority classes.
"""
probs = np.mean( #soft voting (%,%,%,%,%,%,%,%)
    [m.predict_proba(X_test) for m in models],
    axis=0
)
y_pred = np.argmax(probs, axis=1)#averaging
"""
A function was defined to display class prediction
probabilities of the first test sample
as a percentage for individual samples.
"""
def pretty_probs(probs, labels):
    return {labels[i]: f"{round(100*probs[i],2)}%" for i in range(len(labels))}
print("\nExample probabilities for first test sample:")
print(pretty_probs(probs[0], class_labels))




#-----------------------------------------------------
# Evaluation metrics
#-----------------------------------------------------
print("\n\n============ Evaluation metrics... ============")
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, probs)
macro_f1 = f1_score(y_test, y_pred, average='macro')
macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
print(f"Accuracy: {acc:.4f}")
print(f"Log-loss: {ll:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Macro-Precision: {macro_precision:.4f}")
print(f"Macro-Recall   : {macro_recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels))
# Ako vidiš da model sve gura u jednu klasu → nebalansirano
import seaborn as sns
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_optimized.png"), dpi=150, bbox_inches="tight")
plt.show()

# ===== Bootstrap intervali pouzdanosti (test skup = 29 studenata) =====
BOOT_N = 2000
_boot_rng = np.random.default_rng(RANDOM_STATE)

def _metrics_from(y_true_arr, y_pred_arr, y_proba_arr):
    return {
        "accuracy":  accuracy_score(y_true_arr, y_pred_arr),
        "log_loss":  log_loss(y_true_arr, y_proba_arr, labels=list(range(len(class_labels)))),
        "macro_f1":  f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "macro_recall":    recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
    }

def bootstrap_ci(y_true_arr, y_pred_arr, y_proba_arr, n=BOOT_N):
    y_true_arr = np.asarray(y_true_arr); y_pred_arr = np.asarray(y_pred_arr).ravel()
    y_proba_arr = np.asarray(y_proba_arr)
    keys = ["accuracy", "log_loss", "macro_f1", "macro_precision", "macro_recall"]
    acc = {k: [] for k in keys}
    m = len(y_true_arr)
    for _ in range(n):
        idx = _boot_rng.integers(0, m, m)
        if len(np.unique(y_true_arr[idx])) < 2:   # degenerate resample -> skip
            continue
        r = _metrics_from(y_true_arr[idx], y_pred_arr[idx], y_proba_arr[idx])
        for k in keys:
            acc[k].append(r[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 50), np.percentile(v, 97.5)) for k, v in acc.items()}

def print_ci(title, ci):
    print("\n--- {} ---".format(title))
    for k, (lo, med, hi) in ci.items():
        print("  {:<16}: {:.4f}   95% CI [{:.4f}, {:.4f}]".format(k, med, lo, hi))

y_pred_opt = np.asarray(y_pred).ravel().copy()
probs_opt = np.asarray(probs).copy()

print("\n\n============ Bootstrap 95% CI - optimizovani ansambl ============")
ci_opt = bootstrap_ci(y_test, y_pred_opt, probs_opt)
print_ci("Optimizovani model (soft-voting ansambl)", ci_opt)




#-----------------------------------------------------
# SHAP-based feature importance
#-----------------------------------------------------
print("\n\n============ SHAP... ============")

"""
SHAP values and RandomForest's built-in feature importance were used to
interpret the model's predictions, exactly as CatBoost's own SHAP +
FeatureImportance were used in the CatBoost script. shap.TreeExplainer
supports scikit-learn's RandomForestClassifier natively (it walks the
ensemble's decision trees directly, same family of algorithm as
TreeExplainer uses for CatBoost).

Because every feature was one-hot encoded, a single original question (e.g.
column "17") becomes several 0/1 dummy columns (one per answer option). SHAP
values and feature_importances_ are first computed in that expanded one-hot
space, then SUMMED back per original question so the importance table/plots
below are directly comparable to CatBoost's (which reports importance per
original categorical column, not per category level).

As with the CatBoost script, this is averaged across all 5 ensemble members
so it explains the final soft-voting predictor, not a single seed.
"""
n_features = len(feature_names)

_per_model_sv = []          # po-model (n_samples, n_classes, n_features) SHAP nizovi (agregirano na originalne kolone)
_per_model_mean_shap = []
_per_model_mean_abs_shap = []
_per_model_cb = []
for _m in models:
    _prep = _m.named_steps['prep']
    _ohe = _prep.named_transformers_['cat']
    _clf = _m.named_steps['clf']
    _X_train_ohe = _prep.transform(X_train)  # (n_samples, n_features_ohe), dense 0/1 array

    _explainer = shap.TreeExplainer(_clf)
    _raw = _explainer.shap_values(_X_train_ohe)
    # shap's return shape for multiclass tree models varies by version:
    # a list of n_classes (n_samples, n_features_ohe) arrays (older shap), or a
    # single (n_samples, n_features_ohe, n_classes) ndarray (newer shap).
    if isinstance(_raw, list):
        _sv_ohe = np.stack(_raw, axis=1)  # (n_samples, n_classes, n_features_ohe)
    else:
        _raw = np.asarray(_raw)
        if _raw.ndim == 3:
            _sv_ohe = np.transpose(_raw, (0, 2, 1))  # (n_samples, n_classes, n_features_ohe)
        else:
            _sv_ohe = _raw[:, None, :]  # single-output edge case -> add a class axis

    # built-in importance analog: RandomForest's own feature_importances_, in one-hot feature space
    _cbr_ohe = np.asarray(_clf.feature_importances_)  # (n_features_ohe,)

    # map each one-hot output column back to its original categorical feature and sum
    _group_sizes = [len(cats) for cats in _ohe.categories_]  # in cat_features order == feature_names order
    _feature_groups = np.repeat(np.arange(len(_group_sizes)), _group_sizes)

    _sv = np.zeros(_sv_ohe.shape[:2] + (n_features,))
    for _j in range(_sv_ohe.shape[-1]):
        _sv[:, :, _feature_groups[_j]] += _sv_ohe[:, :, _j]
    _cbr = np.zeros(n_features)
    np.add.at(_cbr, _feature_groups, _cbr_ohe)

    _per_model_sv.append(_sv)
    _per_model_mean_shap.append(np.mean(_sv, axis=(0, 1)))
    _per_model_mean_abs_shap.append(np.mean(np.abs(_sv), axis=(0, 1)))
    _per_model_cb.append(_cbr)

# usrednjeno preko 5 modela = objasnjenje finalnog (soft-voting) prediktora
mean_shap = np.mean(_per_model_mean_shap, axis=0)
mean_abs_shap = np.mean(_per_model_mean_abs_shap, axis=0)
cb_importance = np.mean(_per_model_cb, axis=0)
# standardna devijacija preko 5 modela = mera stabilnosti vaznosti unutar ansambla
mean_abs_shap_std = np.std(_per_model_mean_abs_shap, axis=0)
cb_importance_std = np.std(_per_model_cb, axis=0)
# usrednjeni po-uzorak SHAP niz za dependence/histogram grafike nize
shap_vals_no_base = np.mean(_per_model_sv, axis=0)

_stab = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap,
    'mean_abs_shap_std_across_5': mean_abs_shap_std,
    'cb_importance': cb_importance,
    'cb_importance_std_across_5': cb_importance_std,
}).sort_values('mean_abs_shap', ascending=False)
print("\nVaznost obelezja usrednjena preko 5 modela (+ st. devijacija preko 5):")
print(_stab.head(12).to_string(index=False))

fi_df_mas = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap,
    'mean_abs_shap': mean_abs_shap,
    'cb_importance': cb_importance
}).sort_values('mean_abs_shap', ascending=False)

fi_df_ms = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap,
    'mean_abs_shap': mean_abs_shap,
    'cb_importance': cb_importance
}).sort_values('mean_shap', ascending=False)

fi_df_cb = pd.DataFrame({
    'feature': feature_names,
    'mean_shap': mean_shap,
    'mean_abs_shap': mean_abs_shap,
    'cb_importance': cb_importance
}).sort_values('cb_importance', ascending=False)

print(f"\nAll features influencing {target_col} - sort(mas):")
print(fi_df_mas)
print(f"\nAll features influencing {target_col} - sort(ms):")
print(fi_df_ms)
print(f"\nAll features influencing {target_col} - sort(cb):")
print(fi_df_cb)

#mas
plt.figure(figsize=(8,10))
plt.barh(fi_df_mas.feature, fi_df_mas.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.barh(fi_df_mas.feature, fi_df_mas.mean_shap, alpha=0.6, label='Mean SHAP')
plt.barh(fi_df_mas.feature, fi_df_mas.cb_importance, alpha=0.4, label='RandomForest Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_mas.png"), dpi=150, bbox_inches="tight")
plt.show()
#ma
plt.figure(figsize=(8,10))
plt.barh(fi_df_ms.feature, fi_df_ms.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.barh(fi_df_ms.feature, fi_df_ms.mean_shap, alpha=0.6, label='Mean SHAP')
plt.barh(fi_df_ms.feature, fi_df_ms.cb_importance, alpha=0.4, label='RandomForest Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_ms.png"), dpi=150, bbox_inches="tight")
plt.show()
#cb
plt.figure(figsize=(8,10))
plt.barh(fi_df_cb.feature, fi_df_cb.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.barh(fi_df_cb.feature, fi_df_cb.mean_shap, alpha=0.6, label='Mean SHAP')
plt.barh(fi_df_cb.feature, fi_df_cb.cb_importance, alpha=0.4, label='RandomForest Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_cb.png"), dpi=150, bbox_inches="tight")
plt.show()

#-----separate----
plt.figure(figsize=(8,10))
plt.barh(fi_df_mas.feature, fi_df_mas.mean_abs_shap, alpha=0.6, label='Mean |SHAP|')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_mas_only.png"), dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,10))
plt.barh(fi_df_ms.feature, fi_df_ms.mean_shap, alpha=0.6, label='Mean SHAP')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_ms_only.png"), dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,10))
plt.barh(fi_df_cb.feature, fi_df_cb.cb_importance, alpha=0.4, label='RandomForest Importance')
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance")
plt.title(f"Influential Features for {target_col}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_cb_only.png"), dpi=150, bbox_inches="tight")
plt.show()




#To understand how weekly study hours (column "17") influence predictions:

class_idx = 1  # positive class
shap_class = shap_vals_no_base[:, class_idx, :]  # (n_samples, n_features), aggregated back to original columns
study_hours_idx = feature_names.index("17")

#dependence-style view: raw answer value (category code, as given in DATA.csv) vs aggregated SHAP
plt.figure()
plt.scatter(X_train["17"].astype(int), shap_class[:, study_hours_idx], alpha=0.6)
plt.title("SHAP dependence - weekly study hours (column 17, class 1)")
plt.xlabel("Column 17 answer code")
plt.ylabel("Aggregated SHAP value (summed across one-hot levels)")
plt.savefig(os.path.join(FIGURES_DIR, "shap_dependence_col17.png"), dpi=150, bbox_inches="tight")
plt.show()

#or check distribution:
plt.hist(shap_class[:, study_hours_idx], bins=50)
plt.title("SHAP value distribution for weekly study hours - column 17 (class 1)")
plt.xlabel("SHAP value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(FIGURES_DIR, "shap_histogram_col17.png"), dpi=150, bbox_inches="tight")
plt.show()




"""
 The default RandomForestClassifier model, without hyperparameter tuning or class re-balancing,
 was run to establish a baseline for PSO and optimized model evaluation.
 The model was trained using the same data split and metrics.
 It uses the same one-hot Pipeline setup as the optimized model so that categorical
 encoding is handled identically and the only real differences between baseline and
 optimized model are PSO tuning, class balancing, and the 5-seed ensemble.
"""

# ---------------------------
# Default RandomForest Baseline
# ---------------------------
baseline_model = make_rf_pipeline(random_state=RANDOM_STATE)  # sklearn defaults: n_estimators=100, no max_depth cap, no class_weight
baseline_model.fit(X_train, y_train) # Train
y_pred = baseline_model.predict(X_test) # Predict
baseline_probs = baseline_model.predict_proba(X_test) # needed for its own log-loss

# ---------------------------
# Evaluation Metrics
# ---------------------------
print("\n=== Default RandomForest Baseline ===")
acc = accuracy_score(y_test, y_pred)
ll = log_loss(y_test, baseline_probs)
macro_f1 = f1_score(y_test, y_pred, average="macro")
macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
print(f"Accuracy: {acc:.4f}")
print(f"***Log-loss: {ll:.4f}")
print(f"Macro-F1       : {macro_f1:.4f}")
print(f"Macro-Precision: {macro_precision:.4f}")
print(f"Macro-Recall   : {macro_recall:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n=== Confusion Matrix: ===")
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_baseline.png"), dpi=150, bbox_inches="tight")
plt.show()

# ===== bootstrap CI za bazni model + uparena razlika optimizovani - bazni =====
baseline_pred = np.asarray(y_pred).ravel().copy()
baseline_proba = np.asarray(baseline_probs).copy()
print("\n\n============ Bootstrap 95% CI - bazni (default) model ============")
ci_base = bootstrap_ci(y_test, baseline_pred, baseline_proba)
print_ci("Bazni model (default RandomForest)", ci_base)

print("\n--- Uparena bootstrap razlika (optimizovani - bazni), isti resemplovani test skup ---")
_yt = np.asarray(y_test)
_keys = ["accuracy", "log_loss", "macro_f1", "macro_precision", "macro_recall"]
_pair_rng = np.random.default_rng(RANDOM_STATE)
_diffs = {k: [] for k in _keys}
_m = len(_yt)
for _ in range(BOOT_N):
    _idx = _pair_rng.integers(0, _m, _m)
    if len(np.unique(_yt[_idx])) < 2:
        continue
    _ro = _metrics_from(_yt[_idx], y_pred_opt[_idx], probs_opt[_idx])
    _rb = _metrics_from(_yt[_idx], baseline_pred[_idx], baseline_proba[_idx])
    for _k in _keys:
        _diffs[_k].append(_ro[_k] - _rb[_k])
for _k in _keys:
    _d = np.array(_diffs[_k])
    _better = np.mean(_d < 0) if _k == "log_loss" else np.mean(_d > 0)
    print("  {:<16}: razlika median {:+.4f}   95% CI [{:+.4f}, {:+.4f}]   udeo resemplova u korist optimizovanog: {:.1%}".format(
        _k, np.percentile(_d, 50), np.percentile(_d, 2.5), np.percentile(_d, 97.5), _better))
print("NAPOMENA: ovo NIJE formalni test znacajnosti; sluzi samo kao gruba mera pouzdanosti razlike "
      "na test skupu od svega 29 studenata.")


# ===== Ablaciona analiza - koja komponenta donosi poboljsanje =====
# Ablacija dodaje komponente jednu po jednu:
#   A) default RandomForest                     (= bazni model)
#   B) + balansirane tezine klasa
#   C) + PSO hiperparametri (jedan model)
#   D) + ansambl 5 modela sa soft-voting-om     (= optimizovani model)
print("\n\n============ Ablaciona analiza ============")

def _eval_block(name, y_true_arr, y_pred_arr, y_proba_arr):
    r = _metrics_from(np.asarray(y_true_arr), np.asarray(y_pred_arr).ravel(), np.asarray(y_proba_arr))
    print("  {:<58} acc={:.4f}  logloss={:.4f}  macroF1={:.4f}  macroP={:.4f}  macroR={:.4f}".format(
        name, r["accuracy"], r["log_loss"], r["macro_f1"], r["macro_precision"], r["macro_recall"]))
    return r

# A) default RandomForest (bez tezina, bez PSO, bez ansambla)
_A = make_rf_pipeline(random_state=RANDOM_STATE)
_A.fit(X_train, y_train)
_eval_block("A) default RandomForest (= bazni)", y_test,
            _A.predict(X_test), _A.predict_proba(X_test))

# B) + balansirane tezine klasa
_B = make_rf_pipeline(class_weight=class_weight_dict, random_state=RANDOM_STATE)
_B.fit(X_train, y_train)
_eval_block("B) + balansirane tezine klasa", y_test,
            _B.predict(X_test), _B.predict_proba(X_test))

# C) + PSO hiperparametri (jedan model)
_C = make_rf_pipeline(
    n_estimators=1000,
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    class_weight=class_weight_dict,
    random_state=0
)
_C.fit(X_train, y_train)
_eval_block("C) + PSO hiperparametri (jedan model)", y_test,
            _C.predict(X_test), _C.predict_proba(X_test))

# D) + ansambl 5 modela sa soft-voting-om (= optimizovani model, vec izracunat gore)
_eval_block("D) + ansambl 5 modela, soft voting (= optimizovani)", y_test, y_pred_opt, probs_opt)
print("NAPOMENA: PSO hiperparametri (best_params) su nasledjeni iz gornje optimizacije; "
      "ablacija ne pokrece PSO ponovo.")




#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Data verification from AI project
#DataFrame object
print(type(df))
print("Loaded:", df.shape)
print("Columns:", list(df.columns))
#uzorak data seta:
print("head: \n{}\n".format(df_clean.head())) #print(df_clean[0:5]) #ispisi prvih 5 redova iz dataframe-a
print("tail: \n{}\n".format(df_clean.tail)) #ispisivanje poslednjih 5 redova
#indeksi pocinju od 0, korak je 1,
print("index: {}\n".format(df_clean.index))
print("1. red dataframe-a: \n{}\n".format(df_clean.iloc[0])) #1. red dataframe-a
#dimenzija dataframe-a (2D):
print("DataFrame ima %d dimenzije" % df_clean.ndim)
#ispisi featur-e/kolone, target-e/labele, vrednosti
print("Featurs/Kolone: ", df_clean.columns)
print(f"Odgovori za {target_col} (target/label variable): ", df_clean[target_col].unique())
print("Vrednosti: \n", df_clean.values) #prosledjuje samo vrednosti
#koliko ima redova i kolona
(row, col) = df_clean.shape
print("Dataframe ima %d zapisa/redova i %d obelezja/kolona(labels)\n" % (row, col))
print("Dataframe ima %d zapisa/redova i %d obelezja/kolona(labels)\n" % (len(df_clean), len(df_clean.columns)))
#transponovan dataframe
print("Transponovan dataframe: \n{}".format(df_clean.T))
#sve max i min vrednosti kolona
max = df_clean.apply(np.max)
min = df_clean.apply(np.min)
print("max vrednosti kolona: \n{}\n".format(max))
print("min vrednosti kolona: \n{}\n".format(min))
print(df_clean.info())
print("(.info) --- Zakljucak:\nDataFrame ima %d redova i %d kolona\nVidimo kako su imenovane kolone\nKolone imaju int podatke (int64)\nsvaka kolona ima non-null vrednosti, tj. nemamo nedostajuce vrednosti\n" % (row, col))
print(df_clean.describe()) #osnovne statisticke podatke
print("(.describe) --- Zakljucak:\ncount: broj non-null podataka za svaku kolonu, nemamo prazna polja\nmean(avg): srednja vrednost svakog pitanja\nstd(standardna devijacija): disperzija vrednosti svake kolone oko mean, veci std ukazuje na vecu disperziju oko mean vrednosti\nmin,max: najmanja i najveca vrednost za svaku kolonu\n%: distribucija podataka, npr 25% ispitanika je izabralo odredjenu vrednost za odredjeno pitanje\n")
print(df_clean.corr())
print(f"(.corr) --- Zakljucak:\nprocena stepena linearnog odnosa izmedju 2 numbericke variable\nblizu 1: jaka pozitivna linearna veza, v1 se povecava kako se v2 povecava\nblizu-1: jaka negativna linearna veza, v1 se povecava kako se v2 povecava\nblizu 0: slaba ili nikakva linearna veza\npogledati korelacije kolona sa {target_col}\n")
