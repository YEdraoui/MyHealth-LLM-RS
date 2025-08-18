import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def safe_auroc(y_true, y_prob):
    # returns None if only one class present
    vals=[]
    for j in range(y_true.shape[1]):
        yt=y_true[:,j]; yp=y_prob[:,j]
        if len(set(yt))>1:
            try: vals.append(roc_auc_score(yt, yp))
            except Exception: vals.append(np.nan)
        else:
            vals.append(np.nan)
    return vals

def safe_aupr(y_true, y_prob):
    vals=[]
    for j in range(y_true.shape[1]):
        yt=y_true[:,j]; yp=y_prob[:,j]
        if len(set(yt))>1:
            try: vals.append(average_precision_score(yt, yp))
            except Exception: vals.append(np.nan)
        else:
            vals.append(np.nan)
    return vals

def summarize_metrics(y_true, y_prob, label_names):
    auroc = np.array(safe_auroc(y_true, y_prob))
    aupr  = np.array(safe_aupr(y_true, y_prob))
    per_label_auroc = {label_names[j]: (None if np.isnan(auroc[j]) else float(auroc[j])) for j in range(len(label_names))}
    per_label_aupr  = {label_names[j]: (None if np.isnan(aupr[j])  else float(aupr[j]))  for j in range(len(label_names))}
    macro_auroc = float(np.nanmean(auroc)) if np.isfinite(np.nanmean(auroc)) else None
    macro_aupr  = float(np.nanmean(aupr))  if np.isfinite(np.nanmean(aupr))  else None
    # prevalence-weighted AUROC
    prev = np.clip(np.mean(y_true, axis=0), 1e-8, 1-1e-8)
    weights = prev/prev.sum()
    pw_auroc = None
    if np.all(np.isnan(auroc)):
        pw_auroc=None
    else:
        au = np.where(np.isnan(auroc), 0.0, auroc)
        pw_auroc = float(np.sum(au*weights))
    return dict(macro_auROC=macro_auroc, macro_AUPR=macro_aupr, prev_weighted_AUROC=pw_auroc,
                per_label_AUROC=per_label_auroc, per_label_AUPR=per_label_aupr)
