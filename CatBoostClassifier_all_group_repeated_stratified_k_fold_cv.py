# ========================================
#  Higher Education Students Performance – CatBoostClassifier model, Particle Swarm Optimization, and SHAP
#  Predictive and Interpretable Machine Learning Model for Modeling Students' Final Grade
# ========================================
"""
This research aims to share insights and knowledge regarding the development of a
predictive and interpretable machine learning model, CatBoostClassifier,
for predicting students' end-of-term final grades and
explaining the model's predictions employing SHAP, an explainable AI method.
As well as share the process of tuning CatBoostClassifier hyperparameters using
Particle Swarm Optimization (PSO), a population-based metaheuristic algorithm,
with the following design:
  1) 80% train / 20% test split.
  2) On the 80% train set, PSO searches for hyperparameters using a CatBoostClassifier fitness
     function evaluated via 3-fold REPEATED stratified cross-validation (with class_weights). The
     final hyperparameters PSO settles on are then re-evaluated across those same repeated-CV folds
     (this time computing the full metric set, not just the macro-F1 PSO used internally), and the
     result is reported as mean +/- std and a 95% confidence interval across the folds.
  3) Using that final set of hyperparameters, 5 independent CatBoostClassifier "replicates" are
     trained, each as its own 3-fold REPEATED stratified cross-validation ensemble (9 fold-models,
     with class_weights) on the 80% train set; every fold-model predicts on the untouched 20% test
     set and all predictions are soft-voted together. Mean +/- std and a 95% confidence interval are
     reported both across the 5 replicates and (via bootstrap resampling) for the final prediction.

The target GRADE is GROUPED into 3 ordered categories: Low = grades 0,1,2 (67 students),
Medium = grades 3,4,5 (48 students), High = grades 6,7 (30 students). Grouping reduces the number of
target values from 8 to 3 (the rarest original grade had only 8 students) and the imbalance from
4.375x to about 2.2x (Low vs. High).

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
(personal questions, family questions, and education habit questions)
are used as categorical features. STUDENT ID was dropped: it is
unique per row (145 distinct values for 145 students), so it carries no
generalizable signal and would only add noise/overfitting risk.
COURSE ID is also dropped: it dominated the SHAP / CatBoost importance (the model mostly learned
'which course is it' and the typical grade distribution of that course, not properties of the student),
so this version uses only the 30 student questionnaire features (personal, family, education habits).

Symmetries & Asymmetries: Although the questionnaire exhibited structural symmetry in questionnaire design
through questions with a predefined set of answers, the response distribution for
question GRADE showed strong asymmetry, with the highest-engagement
original category '1' being 4.375 times more than the lowest-engagement original category '0'.
This was heavily imbalanced (8 original grades), which is why the grades are grouped into Low/Medium/High.
The distribution of the GRADE target column is checked
below via a value count. Class imbalance across the eight original grade categories is
expected, since some grades are naturally awarded far more often than others,
which can result in biased predictions if left unaddressed.

All stages of the study, including data loading, preprocessing,
model optimization, training, evaluation metrics, and interpretability,
were implemented in Python and executed in the Google Colab environment.

============================================================================
IMPORTANT - COMPUTATIONAL COST WARNING
============================================================================
Step 2 runs a single PSO search (10 particles x 15 iterations), where every
particle evaluation trains 3 x 3 = 9 CatBoost models (3-fold stratified CV
repeated 3 times). A single, non-nested PSO run with this exact fitness
design previously took about 9-10 hours. Step 3 then trains 5 x 9 = 45 more
CatBoost models for the final ensemble, and SHAP explains all 45 of them
(much slower than explaining 5 models), plus the ablation study adds close to
another 20 fits on top of that. Expect the full script to take somewhere in
the order of half a day. PSO_N_PARTICLES / PSO_N_ITERS (defined near the top
of Step 2) and n_repeats in the RepeatedStratifiedKFold calls can be lowered
for a quick smoke test.
============================================================================
"""



#--------------------------------------Installing & importing libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
import matplotlib.pyplot as plt
import pyswarms as ps
from joblib import Parallel, delayed

print("\n\n============ PREPROCESSING... ============")
print("- The dataset is read from the CSV file and stored as a DataFrame object.")
print("- STUDENT ID column is dropped (unique per row, no generalizable signal).")
print("- COURSE ID column is dropped (it dominated the importance: the model learned the course, not the student).")
print("- The categorical target is column 'GRADE', while all remaining columns are treated as categorical features.")
DATA_PATH = "DATA.csv"
df = pd.read_csv(DATA_PATH)
df_clean = df.drop(columns=['STUDENT ID', 'COURSE ID'])   # COURSE ID removed: only student questionnaire features (1-30) remain
target_col = 'GRADE'
#This will show True in any column where hidden whitespace exists:
#(df_clean.astype(str) != df_clean.astype(str).apply(lambda col: col.str.strip())).any()
#if everything is false:
X = df_clean.drop(columns=[target_col]).astype(str)  #features
y = df_clean[target_col].astype(str)                 #target
#if something is true:
#X = df_clean.drop(columns=[target_col]).astype(str).applymap(lambda v: v.strip())
#y = df_clean[target_col].astype(str).apply(lambda v: v.strip())
feature_names = list(X.columns); print("Features names (X):", feature_names)
cat_features = list(range(X.shape[1])); print("***Categorical features names:", cat_features)
print("- Target groups are encoded as integers 0/1/2 (Low/Medium/High) suitable for 3-class classification.")
GRADE_TO_GROUP = {0: 0, 1: 0, 2: 0,   3: 1, 4: 1, 5: 1,   6: 2, 7: 2}   # Low, Medium, High
class_labels = ['Low (0-2)', 'Medium (3-5)', 'High (6-7)']
y_enc = df_clean[target_col].map(GRADE_TO_GROUP).to_numpy(dtype=int)             # 0 = Low, 1 = Medium, 2 = High
print("Target name (y):", target_col)
print("Encoded target values (0/1/2):", class_labels)
print("Original grade value counts: \n{}\n".format(df_clean[target_col].value_counts().sort_index()))
print("Grouped target value counts: \n{}\n".format(pd.Series(y_enc).map(dict(enumerate(class_labels))).value_counts()))
print("--- Balanced:             class with highest value count <= 2-2,5 times class with lowest value count; classes have similar value counts")
print("--- Moderately balanced:  class with highest value count > 3-4 times class with lowest value count")
print("--- Severely imbalanced:  class with highest value count == 5-10 times class with lowest value count; class has < 20-30 value count")
print("- Conclusion: Moderately balanced groups, 'Low' (67) is about 2.2 times more than 'High' (30).")
print("- Balanced class weights are used in order to mitigate the effects of target class imbalance.")
print("- Balanced class weights are incorporated into the CatBoost training process to ensure the model learns from all classes fairly.")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)
print("Class weights:", class_weights)


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
BOOT_N = 2000
_boot_rng = np.random.default_rng(RANDOM_STATE)
_METRIC_KEYS = ["accuracy", "log_loss", "macro_f1", "macro_precision", "macro_recall"]


print("How reliable are the metrics we got on these 29 test students?")
print("Test set of 29 instances is too small for strong conclusions.")
print("That's why we use Bootstrap 95% confidence interval (2.5 / 50 / 97.5 percentil)")
print("So, before this point, we trained the model, we gave it the test set (29 instances) and got predictions.")
print("Now, we give a bootstrap test set (29 instances but some are repeated). ")
print("We call the model 2000 times with a different bootstrap test set (29 instances but some are repeated)")
print("And show the bootstrap prediction [50] and the 95% interval [2.5, 97.5]")
print("The result could vary if you had a slightly different sample of test students -> Bootstrap gives you an estimate of that uncertainty")

def _metrics_from(y_true_arr, y_pred_arr, y_proba_arr):
    return {
        "accuracy":  accuracy_score(y_true_arr, y_pred_arr),
        "log_loss":  log_loss(y_true_arr, y_proba_arr, labels=list(range(len(class_labels)))),
        "macro_f1":  f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
        "macro_recall":    recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
    }

def bootstrap_ci(y_true_arr, y_pred_arr, y_proba_arr, n=BOOT_N):
    """Resamples the (fixed) test rows n times, with replacement, and returns the 2.5/50/97.5
    percentile of each metric. This estimates how much the point estimate on THIS test set could
    have varied if a slightly different sample of the same 29 test students had been drawn."""
    y_true_arr = np.asarray(y_true_arr); y_pred_arr = np.asarray(y_pred_arr).ravel()
    y_proba_arr = np.asarray(y_proba_arr)
    acc = {k: [] for k in _METRIC_KEYS}
    m = len(y_true_arr)
    for _ in range(n):
        idx = _boot_rng.integers(0, m, m)
        if len(np.unique(y_true_arr[idx])) < 2:   # degenerate resample -> skip
            continue
        r = _metrics_from(y_true_arr[idx], y_pred_arr[idx], y_proba_arr[idx])
        for k in _METRIC_KEYS:
            acc[k].append(r[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 50), np.percentile(v, 97.5)) for k, v in acc.items()}

def print_ci(title, ci):
    print("\n--- {} ---".format(title))
    for k, (lo, med, hi) in ci.items():
        print("  {:<16}: {:.4f}   95% CI [{:.4f}, {:.4f}]".format(k, med, lo, hi))

def mean_std_ci(vals, confidence=0.95):
    """Mean, sample std, and a t-distribution confidence interval for a small sample of point
    estimates (e.g., one score per outer split, or one score per ensemble seed)."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / np.sqrt(n)
        tcrit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
        margin = tcrit * se
    else:
        margin = 0.0
    return mean, std, mean - margin, mean + margin

def print_mean_std_ci(title, records, keys=None):
    keys = keys or _METRIC_KEYS
    print("\n--- {} (n={}) ---".format(title, len(records)))
    for k in keys:
        vals = [r[k] for r in records]
        mean, std, lo, hi = mean_std_ci(vals)
        print("  {:<16}: {:.4f} +/- {:.4f}   95% CI [{:.4f}, {:.4f}]".format(k, mean, std, lo, hi))




print("\n\n============ Step 1: train_test_split (80% / 20%)... ============")
print("Why stratification of the target for train/test split?")
print("--- Split will be 80/20 but that does not guarantee that the target class distribution will be the same in both sets.")
print("--- Stratification ensures that the target class distribution is preserved in both training and test sets.")
print("--- In other words, it will not happen that the test set has all representations of one class and the training set has 0.")
print("--- --- The train_test_split makes a representative 80/20 train/test split.")
print("--- --- The 80% training set is used for the PSO hyperparameter search (Step 2) and the final ensemble (Step 3).")
print("--- --- The 20% test set is touched only once, at the very end of Step 3, for the final evaluation.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=RANDOM_STATE
)
print("training set: {} students | test set: {} students".format(len(X_train), len(X_test)))




print("\n\n============ Step 2: PSO hyperparameter search on the 80% train set (3-fold repeated stratified CV fitness) ============")
print("For hyperparameter optimization, Particle Swarm Optimization (PSO) is used to identify an optimal learning rate, depth, L2 regularization, and random_strength.")
print("--- Imagine it like this: PSO optimizer randomly generates particles within defined bounds.")
print("--- Here, the pso optimizer is implemented using the Global Best topology ('global best pso optimizer').")
print("--- Global best influences particles movement -> particles are influenced to move toward the global-best particle (particle with the best fitnesse).")
print("--- The number of particles(10) and iterations(15) was chosen to balance optimization quality and computational efficiency.")
print("PSO optimizer proposes particles (hyperparameter combinations) and calls a custom fitness function to evaluate them.")
print("Here, the fitness function evaluates all particles in parallel using a temporary CatBoostClassifier with 3-fold REPEATED stratified cross-validation (with class_weights) on the training set only (80%).")
print("--- Why CatBoostClassifier?")
print("--- --- Because it is used for multiclass classification problems")
print("--- Why k-fold stratified cross-validation strategy?")
print("--- --- To get a more reliable particle evaluation score (splitting the training set into 3 folds)")
print("--- --- Why spliting is better?")
print("--- --- --- Because splitting once, the score can depend on which students happen to be in the test set.")
print("--- --- --- When it is split 3 times, the score is more reliable.")
print("--- --- Why it has shuffle enabled?")
print("--- --- --- Because the folds could be affected by the original ordering of the data.")
print("--- --- --- The data will be shuffled before the folds are created")
print("--- --- StratifiedKFold creates folds while maintaining approximately the same class distribution in each fold (like the strytify in train_test_split).")
print("--- --- Why 3-fold?")
print("--- --- --- The 3-fold setup provided a balance between computational efficiency and statistical reliability.")
print("--- Why REPEATED k-fold stratified cross-validation strategy?")
print("--- --- 3 folds x 3 repeats = 9 evaluations per particle give a more stable fitness score than a single 3-fold pass,")
print("--- --- because splitting once means the score can depend heavily on which students land in which fold.")
print("--- Why only on the training set?")
print("--- --- Because the test set is reserved for final evaluation and should not be used during hyperparameter optimization.")
print("--- --- The training set is split into 3 folds and the fitness function evaluation score is based on the train/test set in that split.")
print("The fitness function calculates the Macro F1 score for each of the 9 folds and returns the negative mean Macro F1 score to PSO.")
print("The goal is to find the particle that gives the highest evaluation score (mean Macro F1 score across the 9 folds).")
print("Why?")
print("--- F1 score measures how well the model predicts one class.")
print("--- Macro F1 score measures how well the model predicts each class (average of F1 scores across all classes)")
print("--- The model needs to make good predictions for all classes, not just the majority class.")
print("--- That's why we need to find a particle that gives the highest average Macro F1 score!")
print("--- Macro F1 score tells us about the model's balanced performance across all classes, which is important for imbalanced datasets like this one.")
print("Important thing to note: PSO minimizes the objective/fitness function (in other words, PSO will choose the lowest Macro F1 score returned by the fitness function).")
print("This is why the evaluation score (Macro F1 score) returned to PSO is negated.")
print("--- So, when PSO chooses the lowest negative Macro F1 score its actually choosing the highest positive Macro F1 score.")
print("--- in other words, by minimizing the negative Macro F1 score, PSO is maximizing the positive Macro F1 score.")
print("A fixed random seed (42) was used to ensure reproducibility throughout hyperparameter evaluations.")

PSO_N_PARTICLES = 10
PSO_N_ITERS = 15
print("PSO_N_PARTICLES={}, PSO_N_ITERS={} (see the cost warning in the file docstring)".format(PSO_N_PARTICLES, PSO_N_ITERS))

rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=RANDOM_STATE)  # 9 fold evals for a more stable fitness score

def pso_objective_parallel(particles):  #PSO parallelized objective function (objective==fitness)
    def evaluate_particle(p):
        lr = float(p[0])
        depth = int(round(p[1]))
        l2 = int(round(p[2]))
        random_strength = float(p[3])
        fold_scores = []

        for train_idx, test_idx in rskf.split(X_train, y_train):# ===== Corection-2: only on training set =====
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]

            #pool - CatBoost's optimized data structure used to store features, labels, and categorical features
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            te_pool = Pool(X_te, y_te, cat_features=cat_features)

            model = CatBoostClassifier(
                iterations=500,  # smaller for speed
                learning_rate=lr,
                depth=depth,
                l2_leaf_reg=l2,
                random_strength=random_strength,  #to resist overfitting on this small, noisy dataset
                loss_function='MultiClass',
                eval_metric='MultiClass',
                random_seed=RANDOM_STATE,
                early_stopping_rounds=50,
                verbose=False,
                class_weights=class_weights,
                thread_count=-1,
                allow_writing_files=False
            )
            model.fit(tr_pool, eval_set=te_pool, use_best_model=True)
            preds = np.argmax(model.predict_proba(te_pool), axis=1)
            fold_scores.append(f1_score(y_te, preds, average='macro'))

        return -np.mean(fold_scores)

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
    n_particles=PSO_N_PARTICLES,
    dimensions=4,
    options={'c1': 1.4, 'c2': 1.4, 'w': 0.7},
    bounds=bounds
)
cost, pos = optimizer.optimize(pso_objective_parallel, iters=PSO_N_ITERS) # more thorough search
best_params = {
    'learning_rate': float(pos[0]),
    'depth': int(round(pos[1])),
    'l2_leaf_reg': int(round(pos[2])),
    'random_strength': float(pos[3])
}
# best_params = {
#     'learning_rate': ,
#     'depth': ,
#     'l2_leaf_reg': ,
#     'random_strength': 
# }
print("\nBest parameters found by PSO after {} iterations:".format(PSO_N_ITERS))
print(best_params)

print("\n\n============ Step 2 result: PSO hyperparameters re-evaluated over the 3-fold repeated stratified CV folds ============")
print("PSO's own fitness function only tracked the (negated) mean Macro F1 across the 9 folds. This pass")
print("re-trains on those SAME 9 folds using the final best_params, this time computing the full metric")
print("set (accuracy, log-loss, macro-F1, macro-precision, macro-recall) per fold, so we can report the")
print("mean +/- standard deviation and a 95% confidence interval for PSO's result across the 9 folds.")
cv_fold_metrics = []
for train_idx, test_idx in rskf.split(X_train, y_train):
    X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_tr, y_te = y_train[train_idx], y_train[test_idx]
    fold_model = CatBoostClassifier(
        iterations=500,
        learning_rate=best_params['learning_rate'],
        depth=best_params['depth'],
        l2_leaf_reg=best_params['l2_leaf_reg'],
        random_strength=best_params['random_strength'],
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=RANDOM_STATE,
        early_stopping_rounds=50,
        verbose=False,
        class_weights=class_weights,
        thread_count=-1,
        allow_writing_files=False
    )
    fold_model.fit(Pool(X_tr, y_tr, cat_features=cat_features),
                    eval_set=Pool(X_te, y_te, cat_features=cat_features), use_best_model=True)
    fold_proba = fold_model.predict_proba(Pool(X_te, y_te, cat_features=cat_features))
    fold_pred = np.argmax(fold_proba, axis=1)
    cv_fold_metrics.append(_metrics_from(y_te, fold_pred, fold_proba))

print_mean_std_ci("PSO result: final hyperparameters over the 3-fold repeated stratified CV folds", cv_fold_metrics)




print("\n\n============ Step 3: final ensemble -- 5 x (3-fold repeated stratified CV), trained on 80%, tested on 20% ============")
print("Using the final hyperparameters from Step 2, 5 independent 'replicates' are trained.")
print("--- Each replicate runs its own 3-fold REPEATED stratified cross-validation (3 folds x 3 repeats")
print("--- = 9 fold-models, with class_weights) over the WHOLE 80% training set: every fold-model is")
print("--- trained on that fold's training portion (with the held-out fold as eval_set for early")
print("--- stopping) and then predicts on the untouched 20% test set.")
print("--- The 9 fold-models of one replicate are soft-voted (averaged) into that replicate's prediction.")
print("--- The 5 replicates are then soft-voted together -- 45 CatBoost models in total contribute to")
print("--- the final prediction, which is exactly the kind of full-ensemble explanation SHAP uses below.")

seed_level_metrics = []   # one metrics dict per replicate (seed), from that replicate's own averaged prediction
seed_level_probs = []     # one (n_test, n_classes) array per replicate
models = []                # every one of the 45 fold-models, kept for the SHAP section below

for seed in [0, 1, 2, 3, 4]:
    print("\n--- replicate (seed) {} ---".format(seed))
    seed_rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=seed)
    fold_probs = []
    for fold_i, (tr_idx, val_idx) in enumerate(seed_rskf.split(X_train, y_train)): # ===== Corection-2: only on training set =====
        X_tr, X_val_fold = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val_fold = y_train[tr_idx], y_train[val_idx]
        m = CatBoostClassifier(
            iterations=2000,
            learning_rate=best_params['learning_rate'],
            depth=best_params['depth'],
            l2_leaf_reg=best_params['l2_leaf_reg'],
            random_strength=best_params['random_strength'],
            loss_function='MultiClass',
            eval_metric='MultiClass',
            random_seed=seed,
            early_stopping_rounds=100,
            verbose=False,
            class_weights=class_weights,
            allow_writing_files=False
        )
        m.fit(
            Pool(X_tr, y_tr, cat_features=cat_features),
            eval_set=Pool(X_val_fold, y_val_fold, cat_features=cat_features),
            use_best_model=True
        )
        models.append(m)
        fold_probs.append(m.predict_proba(Pool(X_test, y_test, cat_features=cat_features)))
    seed_probs = np.mean(fold_probs, axis=0)   # soft-vote over this replicate's 9 fold-models
    seed_pred = np.argmax(seed_probs, axis=1)
    seed_metrics = _metrics_from(y_test, seed_pred, seed_probs)
    print("  replicate {} metrics (9 fold-models soft-voted, evaluated on the 20% test set):".format(seed),
          {k: round(v, 4) for k, v in seed_metrics.items()})
    seed_level_metrics.append(seed_metrics)
    seed_level_probs.append(seed_probs)

probs = np.mean(seed_level_probs, axis=0)   # soft-vote across the 5 replicates (= soft-vote of all 45 models)
def pretty_probs(probs, labels):
    return {labels[i]: f"{round(100*probs[i],2)}%" for i in range(len(labels))}
print("\nExample probabilities for first test sample: {}".format(pretty_probs(probs[0], class_labels)))
print("Now we have one prediction for every test student but the predition gives (%) for every target value.")
print("Because we want to show 'we predict this grade for this test student', y_pred gives us only the largest (%).")
y_pred = np.argmax(probs, axis=1)

print_mean_std_ci("Ensemble-replicate variability across the 5 seeds (each seed = 9 fold-models soft-voted)", seed_level_metrics)




print("\n\n============ Evaluation metrics... ============")
print("y_test (real test students answers) vs y_pred (predicted test students answers)")
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
print("--- Conclusion: check per-group precision/recall above; the model may still lean on the largest group (Low).")

print("\nConfusion matrix:")
import seaborn as sns
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
print("--- Conclusion: look at which groups the predictions are pushed into (rows = true group, columns = predicted group).")





# saving optimized predictions
y_pred_opt = np.asarray(y_pred).ravel().copy()
probs_opt = np.asarray(probs).copy()

ci_opt = bootstrap_ci(y_test, y_pred_opt, probs_opt)
print_ci("Bootstrap 95% CI Optimized model (soft-voting ansambl, 45 modela)", ci_opt) # ===== Corection-1: Bootstrap 95% CI Optimized model =====




print("\n\n============ SHAP...CatBoost's built-in feature importance... ============")
print("Interpreting the predictions of all {} models that contribute to the final soft-voted prediction".format(len(models)))
print("Multi-class output was handled by producing one global importance value per feature.")
print("--- SHAP importance")
print("Because it is model-specific, SHAP importance (mean SHAP and mean |SHAP|) was calculated for each model")
print("The mean |SHAP| values were calculated across all samples and classes to obtain global feature importance scores.")
print("After that, it calculates the mean values across all {} models".format(len(models)))
print("--- CatBoost's built-in feature importance")
print("Because it is model-specific, CatBoost's built-in feature importance was calculated for each model")
print("After that, it calculates the mean values across all {} models".format(len(models)))
print("NOTE: the final ensemble has {} models (5 replicates x 9 CV folds) instead of just 5, so this step".format(len(models)))
print("is proportionally slower than explaining a plain 5-model ensemble.")

train_pool = Pool(X_train, y_train, cat_features=cat_features)
n_features = len(feature_names)

_per_model_sv = []          # (n_samples, n_classes, n_features) SHAP arrays per model
_per_model_mean_shap = []
_per_model_mean_abs_shap = []
_per_model_cb = []
for _m in models:
    _sv = np.asarray(_m.get_feature_importance(train_pool, type='ShapValues'))[:, :, :-1]  # bez baseline stuba
    _per_model_sv.append(_sv)
    _per_model_mean_shap.append(np.mean(_sv, axis=(0, 1)))
    _per_model_mean_abs_shap.append(np.mean(np.abs(_sv), axis=(0, 1)))
    _cbr = np.array(_m.get_feature_importance(train_pool, type='FeatureImportance'))
    if len(_cbr) != n_features:
        _nc = len(_cbr) // n_features
        _cbr = _cbr.reshape(_nc, n_features).sum(axis=0)
    _per_model_cb.append(_cbr)

# usrednjeno preko svih modela ansambla = objasnjenje finalnog (soft-voting) prediktora
mean_shap = np.mean(_per_model_mean_shap, axis=0)
mean_abs_shap = np.mean(_per_model_mean_abs_shap, axis=0)
cb_importance = np.mean(_per_model_cb, axis=0)
# standardna devijacija preko svih modela ansambla = mera stabilnosti vaznosti unutar ansambla
mean_abs_shap_std = np.std(_per_model_mean_abs_shap, axis=0)
cb_importance_std = np.std(_per_model_cb, axis=0)
# usrednjeni po-uzorak SHAP niz za dependence/histogram grafike nize
shap_vals_no_base = np.mean(_per_model_sv, axis=0)

_stab = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap_across_5': mean_abs_shap,
    'std_of_mean_abs_shap_across_5': mean_abs_shap_std,
    'cb_importance_across_5': cb_importance,
    'std_of_cb_importance_across_5': cb_importance_std,
}).sort_values('mean_abs_shap_across_5', ascending=False)
print("\nThe feature importance table (mean and std, across all {} models):".format(len(models)))
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
class_idx = 1  # class index 1 = 'Medium (3-5)'
shap_class = shap_vals_no_base[:, class_idx, :]  # (n_samples, n_features)
shap.dependence_plot(
    "17",
    shap_class,
    X_train,
    feature_names=feature_names
)
#or check distribution:
study_hours_idx = feature_names.index("17")
plt.hist(shap_class[:, study_hours_idx], bins=50)
plt.title("SHAP value distribution for weekly study hours - column 17 (class Medium)")
plt.xlabel("SHAP value")
plt.ylabel("Frequency")
plt.show()




print("\n=== Default CatBoostClassifier Baseline (baseline for PSO and optimized model evaluation) ===")
print("- differences between baseline and optimized model are PSO tuning, class balancing, and the 45-model ensemble")
baseline_model = CatBoostClassifier(
    random_state=42,
    verbose=0,
    allow_writing_files=False
)
baseline_model.fit(Pool(X_train, y_train, cat_features=cat_features)) # Train
baseline_test_pool = Pool(X_test, y_test, cat_features=cat_features)
y_pred = baseline_model.predict(baseline_test_pool) # Predict
baseline_probs = baseline_model.predict_proba(baseline_test_pool) # needed for its own log-loss

print("\n=== Evaluation Metrics ===")
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
plt.show()


baseline_pred = np.asarray(y_pred).ravel().copy()
baseline_proba = np.asarray(baseline_probs).copy()
ci_base = bootstrap_ci(y_test, baseline_pred, baseline_proba)
print_ci("Bootstrap 95% CI Default model", ci_base) # ===== Corection-1: bootstrap CI for default model (!!! compare optimized vs base bootstrap)=====
print("\n--- Comparison of Optimized and Default Models Using Bootstrap CI on the same resampled test set ---")
_yt = np.asarray(y_test)
_pair_rng = np.random.default_rng(RANDOM_STATE)
_diffs = {k: [] for k in _METRIC_KEYS}
_m = len(_yt)
for _ in range(BOOT_N):
    _idx = _pair_rng.integers(0, _m, _m)
    if len(np.unique(_yt[_idx])) < 2:
        continue
    _ro = _metrics_from(_yt[_idx], y_pred_opt[_idx], probs_opt[_idx])
    _rb = _metrics_from(_yt[_idx], baseline_pred[_idx], baseline_proba[_idx])
    for _k in _METRIC_KEYS:
        _diffs[_k].append(_ro[_k] - _rb[_k])
for _k in _METRIC_KEYS:
    _d = np.array(_diffs[_k])
    # For log loss, lower values indicate better performance.
    # For all other metrics, higher values indicate better performance.
    _better = np.mean(_d < 0) if _k == "log_loss" else np.mean(_d > 0)
    print("  {:<16}: median difference {:+.4f}   95% CI [{:+.4f}, {:+.4f}]   "
        "proportion of resamples favoring the optimized model: {:.1%}".format(
        _k,
        np.percentile(_d, 50),
        np.percentile(_d, 2.5),
          np.percentile(_d, 97.5),
          _better))




print("\n\n============ Incremental ablation study (adding one component at a time) ============")
print("The evaluation is on the same y_test set")
# ===== Corection-3: ablation table (redesigned around the repeated-CV variant) =====
#   A) default CatBoost
#   B) A + class weights
#   C) B + PSO hyperparameters (single model, no CV ensembling)
#   D) C + 3-fold stratified cross-validation (3 fold-models, PSO hyperparameters, soft-voted)
#   E) C + 3-fold REPEATED stratified cross-validation (9 fold-models, PSO hyperparameters, soft-voted)
#   F) E + ensemble of 5 seeds soft-voting (45 fold-models total) -- this IS the final Step 3 model,
#      so row F simply reuses y_pred_opt / probs_opt instead of recomputing it.
def _eval_block(name, y_true_arr, y_pred_arr, y_proba_arr):
    r = _metrics_from(np.asarray(y_true_arr), np.asarray(y_pred_arr).ravel(), np.asarray(y_proba_arr))
    print("  {:<95} acc={:.4f}  logloss={:.4f}  macroF1={:.4f}  macroP={:.4f}  macroR={:.4f}".format(
        name, r["accuracy"], r["log_loss"], r["macro_f1"], r["macro_precision"], r["macro_recall"]))
    return r


def _cv_ensemble_probs(cv, params=None, seed_for_model=RANDOM_STATE):
    """Trains one CatBoost model per CV fold (fold-train / fold-val as eval_set for early stopping),
    predicts on the (fixed) 20% test set with each fold-model, and soft-votes (averages) them.
    params=None uses CatBoost's own defaults (apart from class_weights); params={...} uses the given
    (PSO-found) hyperparameters instead."""
    fold_probs = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        Xtr, Xv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yv = y_train[tr_idx], y_train[val_idx]
        kwargs = dict(
            loss_function='MultiClass', eval_metric='MultiClass',
            random_seed=seed_for_model, early_stopping_rounds=100, verbose=False,
            class_weights=class_weights, allow_writing_files=False
        )
        if params is not None:
            kwargs.update(
                iterations=2000, learning_rate=params['learning_rate'], depth=params['depth'],
                l2_leaf_reg=params['l2_leaf_reg'], random_strength=params['random_strength']
            )
        else:
            kwargs.update(iterations=1000)
        mdl = CatBoostClassifier(**kwargs)
        mdl.fit(Pool(Xtr, ytr, cat_features=cat_features),
                eval_set=Pool(Xv, yv, cat_features=cat_features), use_best_model=True)
        fold_probs.append(mdl.predict_proba(baseline_test_pool))
    return np.mean(fold_probs, axis=0)


# A)
_A = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False)
_A.fit(Pool(X_train, y_train, cat_features=cat_features))
_eval_block("A) default CatBoost", y_test, _A.predict(baseline_test_pool), _A.predict_proba(baseline_test_pool))

# B)
_B = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False, class_weights=class_weights)
_B.fit(Pool(X_train, y_train, cat_features=cat_features))
_eval_block("B) default CatBoost + class weights", y_test, _B.predict(baseline_test_pool), _B.predict_proba(baseline_test_pool))

# C) single model with PSO hyperparameters (its own internal train/eval-set split, used only for
#    this row's early stopping; the fold-based rows below don't need this since each fold provides
#    its own held-out eval_set)
_X_c_fit, _X_c_es, _y_c_fit, _y_c_es = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)
_C = CatBoostClassifier(
    iterations=2000,
    learning_rate=best_params['learning_rate'], depth=best_params['depth'],
    l2_leaf_reg=best_params['l2_leaf_reg'], random_strength=best_params['random_strength'],
    loss_function='MultiClass', eval_metric='MultiClass',
    random_seed=RANDOM_STATE, early_stopping_rounds=100, verbose=False,
    class_weights=class_weights, allow_writing_files=False
)
_C.fit(Pool(_X_c_fit, _y_c_fit, cat_features=cat_features),
       eval_set=Pool(_X_c_es, _y_c_es, cat_features=cat_features), use_best_model=True)
_eval_block("C) default CatBoost + class weights + PSO", y_test, _C.predict(baseline_test_pool), _C.predict_proba(baseline_test_pool))

# D) C + 3-fold stratified cross-validation (3 fold-models, PSO hyperparameters, soft-voted)
_skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
_D_probs = _cv_ensemble_probs(_skf3, params=best_params)
_eval_block("D) default CatBoost + class weights + PSO + 3-fold stratified cross-validation", y_test, np.argmax(_D_probs, axis=1), _D_probs)

# E) C + 3-fold REPEATED stratified cross-validation (9 fold-models, PSO hyperparameters, soft-voted)
_rskf9 = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=RANDOM_STATE)
_E_probs = _cv_ensemble_probs(_rskf9, params=best_params)
_eval_block("E) default CatBoost + class weights + PSO + 3-fold repeated stratified cross-validation", y_test, np.argmax(_E_probs, axis=1), _E_probs)

# F) E + ensemble of 5 seeds soft-voting (45 fold-models total) == the final Step 3 model, so this
#    row reuses y_pred_opt / probs_opt instead of retraining another 45 models from scratch.
_eval_block("F) default CatBoost + class weights + PSO + 3-fold repeated stratified cross-validation + ensemble 5 modela with soft-voting",
            y_test, y_pred_opt, probs_opt)




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
