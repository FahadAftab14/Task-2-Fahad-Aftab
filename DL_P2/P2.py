import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from imblearn.pipeline import Pipeline       
from imblearn.over_sampling import SMOTE

# STEP 0: LOAD CLEANED DATASET (Project 1 output)

print("\nSTEP 0: LOAD & INSPECT CLEANED DATASET\n")

df = pd.read_csv('cleaned_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Shape  : {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nOrderStatus distribution:\n{df['OrderStatus'].value_counts()}")
print(f"\nPaymentMethod distribution:\n{df['PaymentMethod'].value_counts()}")

# STEP 1: ENGINEER FRAUD TARGET

print("\nSTEP 1: ENGINEER FRAUD TARGET VARIABLE\n")

q75_price = df['TotalPrice'].quantile(0.75)
print(f"75th percentile TotalPrice (high-value threshold): {q75_price:.2f}")

# Fraud conditions
is_returned     = df['OrderStatus'] == 'Returned'
is_anon_payment = df['PaymentMethod'].isin(['Gift Card', 'Cash'])
is_high_value   = df['TotalPrice'] > q75_price

# Fraud flag: Returned + (anonymous payment OR high value)
df['IsFraud'] = (is_returned & (is_anon_payment | is_high_value)).astype(int)

n_fraud = df['IsFraud'].sum()
n_legit = (df['IsFraud'] == 0).sum()
fraud_rate = df['IsFraud'].mean()

print(f"\nFraud Flag Distribution:")
print(f"  Legitimate (0) : {n_legit}   ({(1-fraud_rate)*100:.1f}%)")
print(f"  Fraudulent (1) : {n_fraud}  ({fraud_rate*100:.1f}%)")
print(f"\nBusiness logic:")
print(f"  Returned orders with anonymous payment (Gift Card/Cash) OR high")
print(f"  transaction value (> {q75_price:.0f}) — classic return-fraud pattern.")
print(f"\nTRAP #1 — Naive 'predict all legitimate' accuracy: {(1-fraud_rate)*100:.1f}%")
print(f"           {n_fraud} fraud cases missed. This is why Accuracy is discarded.\n")

# STEP 2: FEATURE PREPARATION

print("\nSTEP 2: FEATURE PREPARATION\n")

drop_cols = ['OrderID', 'CustomerID', 'TrackingNumber', 'ShippingAddress',
             'Date', 'IsFraud', 'OrderStatus']

# One-hot encode remaining categoricals
cat_cols = ['Product', 'PaymentMethod', 'ReferralSource', 'CouponCode']
df_model = pd.get_dummies(df.drop(columns=drop_cols), columns=cat_cols,
                          drop_first=False, dtype=int)

X = df_model.values
y = df['IsFraud'].values
feature_names = df_model.columns.tolist()

print(f"Features used    : {len(feature_names)}")
print(f"Feature list     : {feature_names}")
print(f"Dataset shape    : {X.shape}")

# STEP 3: STRATIFIED TRAIN / TEST SPLIT

print("\nSTEP 3: STRATIFIED TRAIN / TEST SPLIT (80 / 20)\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {X_train.shape[0]} rows | Fraud = {y_train.sum()}  ({y_train.mean()*100:.1f}%)")
print(f"Test  : {X_test.shape[0]} rows  | Fraud = {y_test.sum()}   ({y_test.mean()*100:.1f}%)")
print(f"\n Test set SEALED — no scaling, no SMOTE. Real-world imbalance preserved.")

# Show class counts before and after SMOTE (for illustration)
print(f"\nBefore SMOTE => Train: Legit={( y_train==0).sum()}, Fraud={y_train.sum()}")
smote_preview = SMOTE(random_state=42)
scaler_preview = StandardScaler()
X_prev = scaler_preview.fit_transform(X_train)
_, y_prev = smote_preview.fit_resample(X_prev, y_train)
u, c = np.unique(y_prev, return_counts=True)
print(f"After  SMOTE => Train: Legit={c[0]}, Fraud={c[1]}")
print(f" SMOTE synthesises minority samples — never clones.\n")

# STEP 4: BUILD IMBLEARN PIPELINES

print("\nSTEP 4: CONSTRUCT IMBLEARN PIPELINES\n")

# Pipeline A — Logistic Regression (linear boundary, needs scaling)
pipeline_lr = Pipeline(steps=[
    ('scaler',     StandardScaler()),
    ('smote',      SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Pipeline B — Random Forest (non-linear, scale-invariant)
pipeline_rf = Pipeline(steps=[
    ('smote',      SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42))
])

print("Pipeline A (LR): StandardScaler => SMOTE => LogisticRegression")
print("Pipeline B (RF): SMOTE => RandomForestClassifier")
print(" SMOTE executes inside each CV fold — zero leakage during GridSearch.")

# STEP 5: HYPERPARAMETER TUNING (GridSearchCV, scoring='roc_auc')

print("\nSTEP 5: HYPERPARAMETER TUNING — GridSearchCV (roc_auc)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid_lr = {
    'smote__k_neighbors': [3, 5],
    'classifier__C':      [0.01, 0.1, 1.0]
}

param_grid_rf = {
    'smote__k_neighbors':       [3, 5],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth':    [5, 10, None]
}

print("\nTuning Logistic Regression ...")
gs_lr = GridSearchCV(
    pipeline_lr, param_grid_lr,
    scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0
)
gs_lr.fit(X_train, y_train)
print(f"  Best params : {gs_lr.best_params_}")
print(f"  Best CV AUC : {gs_lr.best_score_:.4f}")

print("\nTuning Random Forest ...")
gs_rf = GridSearchCV(
    pipeline_rf, param_grid_rf,
    scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0
)
gs_rf.fit(X_train, y_train)
print(f"  Best params : {gs_rf.best_params_}")
print(f"  Best CV AUC : {gs_rf.best_score_:.4f}\n")

# STEP 6: FINAL EVALUATION ON SEALED TEST SET

print("\nSTEP 6: FINAL EVALUATION ON SEALED TEST SET\n")

def evaluate_model(name, estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]
    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    print(f"\n{'─'*52}")
    print(f"  MODEL : {name}")
    print(f"{'─'*52}")
    print(f"  Precision  : {prec:.4f}  ← When we flag fraud, are we right?")
    print(f"  Recall     : {rec:.4f}  ← Did we catch all the fraud?")
    print(f"  F1-Score   : {f1:.4f}  ← Harmonic mean (Precision & Recall)")
    print(f"  ROC-AUC    : {auc:.4f}  ← Overall class separation capability")
    print(f"  Avg Prec   : {ap:.4f}  ← Area under Precision-Recall curve")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=['Legitimate', 'Fraudulent'],
                                digits=4))
    return y_pred, y_prob, cm, prec, rec, f1, auc, ap

pred_lr, prob_lr, cm_lr, prec_lr, rec_lr, f1_lr, auc_lr, ap_lr = \
    evaluate_model("Logistic Regression (tuned)", gs_lr.best_estimator_, X_test, y_test)

pred_rf, prob_rf, cm_rf, prec_rf, rec_rf, f1_rf, auc_rf, ap_rf = \
    evaluate_model("Random Forest (tuned)", gs_rf.best_estimator_, X_test, y_test)

winner = "Random Forest" if auc_rf > auc_lr else "Logistic Regression"
best_auc = max(auc_rf, auc_lr)
print(f"\n  WINNER : {winner}  (ROC-AUC = {best_auc:.4f})")

# STEP 7: VISUALISATION DASHBOARD

print("Generating dashboard ...")

fig = plt.figure(figsize=(20, 15))
fig.suptitle(
    'Project 2: E-Commerce Fraud Detection Pipeline — Evaluation Dashboard',
    fontsize=14, fontweight='bold', y=0.99
)

# 1 — Class imbalance (raw)
ax1 = fig.add_subplot(3, 4, 1)
ax1.bar(['Legitimate', 'Fraudulent'], [n_legit, n_fraud],
        color=['steelblue', 'tomato'], edgecolor='white')
ax1.set_title('Class Imbalance\n(Engineered Fraud Target)')
ax1.set_ylabel('Count')
for i, n in enumerate([n_legit, n_fraud]):
    ax1.text(i, n + 8, str(n), ha='center', fontsize=10, fontweight='bold')

# 2 — After SMOTE
ax2 = fig.add_subplot(3, 4, 2)
ax2.bar(['Legitimate', 'Fraudulent'], [c[0], c[1]],
        color=['steelblue', 'tomato'], edgecolor='white')
ax2.set_title('After SMOTE\n(Training fold only)')
ax2.set_ylabel('Count')
for i, n in enumerate([c[0], c[1]]):
    ax2.text(i, n + 8, str(n), ha='center', fontsize=10, fontweight='bold')

# 3 — Confusion Matrix LR
ax3 = fig.add_subplot(3, 4, 3)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Pred Legit', 'Pred Fraud'],
            yticklabels=['True Legit', 'True Fraud'],
            annot_kws={'size': 12})
ax3.set_title(f'Confusion Matrix\nLogistic Regression')

# 4 — Confusion Matrix RF
ax4 = fig.add_subplot(3, 4, 4)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', ax=ax4,
            xticklabels=['Pred Legit', 'Pred Fraud'],
            yticklabels=['True Legit', 'True Fraud'],
            annot_kws={'size': 12})
ax4.set_title(f'Confusion Matrix\nRandom Forest')

# 5 — ROC Curves
ax5 = fig.add_subplot(3, 4, 5)
fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
ax5.plot(fpr_lr, tpr_lr, color='steelblue', lw=2, label=f'LR (AUC={auc_lr:.3f})')
ax5.plot(fpr_rf, tpr_rf, color='tomato',    lw=2, label=f'RF (AUC={auc_rf:.3f})')
ax5.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax5.set_xlabel('FPR'); ax5.set_ylabel('TPR')
ax5.set_title('ROC Curves'); ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

# 6 — Precision-Recall Curves
ax6 = fig.add_subplot(3, 4, 6)
pv_lr, rv_lr, _ = precision_recall_curve(y_test, prob_lr)
pv_rf, rv_rf, _ = precision_recall_curve(y_test, prob_rf)
ax6.plot(rv_lr, pv_lr, color='steelblue', lw=2, label=f'LR (AP={ap_lr:.3f})')
ax6.plot(rv_rf, pv_rf, color='tomato',    lw=2, label=f'RF (AP={ap_rf:.3f})')
ax6.axhline(y_test.mean(), color='k', linestyle='--', lw=1, label='Baseline')
ax6.set_xlabel('Recall'); ax6.set_ylabel('Precision')
ax6.set_title('Precision-Recall Curves'); ax6.legend(fontsize=9); ax6.grid(alpha=0.3)

# 7 — Metric comparison bar chart
ax7 = fig.add_subplot(3, 4, 7)
mets = ['Precision', 'Recall', 'F1', 'ROC-AUC']
lv = [prec_lr, rec_lr, f1_lr, auc_lr]
rv_m = [prec_rf, rec_rf, f1_rf, auc_rf]
x_pos = np.arange(4); w = 0.35
ax7.bar(x_pos - w/2, lv, w, label='LR', color='steelblue', edgecolor='white')
ax7.bar(x_pos + w/2, rv_m, w, label='RF', color='tomato',  edgecolor='white')
ax7.set_xticks(x_pos); ax7.set_xticklabels(mets, fontsize=9)
ax7.set_ylim(0, 1.18)
ax7.set_title('Metric Comparison\n(Accuracy discarded)')
ax7.legend(fontsize=9); ax7.grid(axis='y', alpha=0.3)
for i, (l, r) in enumerate(zip(lv, rv_m)):
    ax7.text(i - w/2, l + 0.02, f'{l:.2f}', ha='center', fontsize=7)
    ax7.text(i + w/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=7)

# 8 — Feature importances (RF)
ax8 = fig.add_subplot(3, 4, 8)
rf_clf = gs_rf.best_estimator_.named_steps['classifier']
importances = rf_clf.feature_importances_
top_idx = np.argsort(importances)[-12:]
ax8.barh(np.array(feature_names)[top_idx], importances[top_idx], color='tomato')
ax8.set_title('Top 12 Feature Importances\n(Random Forest)')
ax8.set_xlabel('Importance Score')

# 9 — Threshold tuning LR
ax9 = fig.add_subplot(3, 4, 9)
thresholds = np.linspace(0.01, 0.99, 100)
pt, rt, ft = [], [], []
for t in thresholds:
    pp = (prob_lr >= t).astype(int)
    pt.append(precision_score(y_test, pp, zero_division=0))
    rt.append(recall_score(y_test, pp, zero_division=0))
    ft.append(f1_score(y_test, pp, zero_division=0))
best_t_lr = thresholds[np.argmax(ft)]
ax9.plot(thresholds, pt, color='steelblue', label='Precision')
ax9.plot(thresholds, rt, color='tomato',    label='Recall')
ax9.plot(thresholds, ft, color='green',     label='F1')
ax9.axvline(best_t_lr, color='purple', linestyle='--',
            label=f'Best F1 @ {best_t_lr:.2f}')
ax9.set_xlabel('Decision Threshold')
ax9.set_title('Threshold Tuning (LR)'); ax9.legend(fontsize=8); ax9.grid(alpha=0.3)

# 10 — Threshold tuning RF
ax10 = fig.add_subplot(3, 4, 10)
pt2, rt2, ft2 = [], [], []
for t in thresholds:
    pp = (prob_rf >= t).astype(int)
    pt2.append(precision_score(y_test, pp, zero_division=0))
    rt2.append(recall_score(y_test, pp, zero_division=0))
    ft2.append(f1_score(y_test, pp, zero_division=0))
best_t_rf = thresholds[np.argmax(ft2)]
ax10.plot(thresholds, pt2, color='steelblue', label='Precision')
ax10.plot(thresholds, rt2, color='tomato',    label='Recall')
ax10.plot(thresholds, ft2, color='green',     label='F1')
ax10.axvline(best_t_rf, color='purple', linestyle='--',
             label=f'Best F1 @ {best_t_rf:.2f}')
ax10.set_xlabel('Decision Threshold')
ax10.set_title('Threshold Tuning (RF)'); ax10.legend(fontsize=8); ax10.grid(alpha=0.3)

# 11 — Accuracy trap illustration
ax11 = fig.add_subplot(3, 4, 11)
naive_acc = (y_test == 0).mean() * 100
lr_acc    = ((cm_lr[0,0] + cm_lr[1,1]) / cm_lr.sum()) * 100
rf_acc    = ((cm_rf[0,0] + cm_rf[1,1]) / cm_rf.sum()) * 100
model_labels = ['Naive\n(all 0)', 'Logistic\nRegression', 'Random\nForest']
accs    = [naive_acc,   lr_acc,       rf_acc]
recalls = [0.0,         rec_lr*100,   rec_rf*100]
xb = np.arange(3); wb = 0.35
b1 = ax11.bar(xb - wb/2, accs,    wb, label='Accuracy %',  color='lightcoral', edgecolor='white')
b2 = ax11.bar(xb + wb/2, recalls, wb, label='Recall %',    color='steelblue',  edgecolor='white')
ax11.set_xticks(xb); ax11.set_xticklabels(model_labels, fontsize=8)
ax11.set_ylim(0, 115)
ax11.set_title('Accuracy vs Recall\n(The Accuracy Trap)'); ax11.legend(fontsize=8)
ax11.grid(axis='y', alpha=0.3)
for b, v in zip(b1, accs):
    ax11.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontsize=7)
for b, v in zip(b2, recalls):
    ax11.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontsize=7)

# 12 — Protocol summary
ax12 = fig.add_subplot(3, 4, 12)
ax12.axis('off')
summary = (
    "FRAUD TARGET\n"
    "─" * 22 + "\n"
    "Returned + (AnonPay OR\n"
    "HighValue) → IsFraud=1\n"
    f"141 fraud / 1059 legit\n\n"
    "ZERO-LEAKAGE PROTOCOL\n"
    "─" * 22 + "\n"
    " Split first, SMOTE after\n"
    " imblearn.pipeline used\n"
    " Scaler inside pipeline\n"
    " CV scoring = roc_auc\n"
    " Accuracy DISCARDED\n\n"
    "FINAL TEST RESULTS\n"
    "─" * 22 + "\n"
    f"LR  AUC : {auc_lr:.4f}\n"
    f"RF  AUC : {auc_rf:.4f}\n"
    f"LR  Recall : {rec_lr:.4f}\n"
    f"RF  Recall : {rec_rf:.4f}\n\n"
    f"WINNER: {winner}\n"
    f"AUC   : {best_auc:.4f}"
)
ax12.text(0.03, 0.98, summary, transform=ax12.transAxes,
          fontsize=8.5, va='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fraud_detection_ecommerce_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n Dashboard saved: fraud_detection_ecommerce_dashboard.png")
