import joblib
import os
import json
import numpy as np

W = os.getcwd()
primary = os.path.join(W, 'model_data.pkl')
candidates = sorted([f for f in os.listdir(W) if f.startswith('model_data_candidate_')])
latest_candidate = os.path.join(W, candidates[-1]) if candidates else None

def summarize(path):
    if not path or not os.path.exists(path):
        return {'path': path, 'error': 'missing'}
    md = joblib.load(path)
    out = {'path': os.path.basename(path)}
    # best_cv_r2 if present
    out['best_cv_r2'] = float(md.get('best_cv_r2')) if md.get('best_cv_r2') is not None else None
    # cv_metrics
    out['cv_metrics'] = md.get('cv_metrics', {})
    # per-model stored metrics
    out['lr_metrics'] = md.get('lr_metrics', {})
    out['rf_metrics'] = md.get('rf_metrics', {})
    out['hgb_metrics'] = md.get('hgb_metrics', {})
    # feature columns
    out['n_features'] = len(md.get('feature_columns', []))
    out['feature_columns_sample'] = md.get('feature_columns', [])[:20]
    # RF feature importances (top 10)
    if md.get('rf_model') is not None and hasattr(md['rf_model'], 'feature_importances_'):
        fi = list(md['rf_model'].feature_importances_)
        cols = md.get('feature_columns', [])
        pairs = sorted(list(zip(cols, fi)), key=lambda x: x[1], reverse=True)[:10]
        out['rf_top_features'] = [{'feature': p[0], 'importance': float(p[1])} for p in pairs]
    else:
        out['rf_top_features'] = None
    # LR coefficients (best-effort)
    lr = md.get('lr_model')
    if lr is not None:
        try:
            if hasattr(lr, 'coef_'):
                coefs = np.array(lr.coef_).ravel().tolist()
                cols = md.get('feature_columns', [])
                if len(coefs) == len(cols):
                    pairs = sorted(list(zip(cols, coefs)), key=lambda x: abs(x[1]), reverse=True)[:10]
                    out['lr_top_coefs'] = [{'feature': p[0], 'coef': float(p[1])} for p in pairs]
                else:
                    out['lr_coef_len'] = len(coefs)
            elif hasattr(lr, 'named_steps') and 'lr' in lr.named_steps:
                lr_step = lr.named_steps['lr']
                if hasattr(lr_step, 'coef_'):
                    coefs = np.array(lr_step.coef_).ravel().tolist()
                    cols = md.get('feature_columns', [])
                    if len(coefs) == len(cols):
                        pairs = sorted(list(zip(cols, coefs)), key=lambda x: abs(x[1]), reverse=True)[:10]
                        out['lr_top_coefs'] = [{'feature': p[0], 'coef': float(p[1])} for p in pairs]
        except Exception as e:
            out['lr_top_coefs_error'] = str(e)
    else:
        out['lr_top_coefs'] = None

    return out

res = {'primary': summarize(primary), 'latest_candidate': summarize(latest_candidate)}
print(json.dumps(res, indent=2, ensure_ascii=False))
