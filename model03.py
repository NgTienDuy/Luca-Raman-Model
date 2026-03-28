"""model03.py — MCR-ALS (NMF) + PLSR with LOO-CV"""
import numpy as np, time, os
from sklearn.decomposition import NMF
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from utils import (preprocess_batch, evaluate, print_metrics, save_checkpoint,
                   load_checkpoint, save_results, plot_predictions,
                   plot_predictions_raw, plot_r2_curve, get_logger, dirichlet_normalize)
from config import MODEL_CONFIGS, LABEL_COLS
log=get_logger("model03"); CFG=MODEL_CONFIGS[3]

def _band_int(X,wns,n=8):
    segs=np.array_split(np.arange(len(wns)),n)
    return np.column_stack([X[:,idx].sum(1) for idx in segs]).astype(np.float32)

def run(X_train,X_test,Y_train,Y_test,wavenumbers=None,retrain=False,
        X_test_raw=None,Y_test_raw=None,sid_test_raw=None,**kw):
    log.info("="*60); log.info("Model 3 — MCR-ALS (NMF) + PLSR")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    t0=time.time()
    if wavenumbers is None: wavenumbers=np.linspace(267,2004,X_train.shape[1])
    if not retrain:
        ck=load_checkpoint(3)
        if ck:
            nmf,pls,sc=ck["nmf"],ck["pls"],ck["sc"]
            Xte=preprocess_batch(X_test,"full")
            Xte=np.maximum(Xte-Xte.min(axis=1,keepdims=True),0)
            fte=sc.transform(np.hstack([nmf.transform(Xte),_band_int(Xte,wavenumbers)]))
            Yp=dirichlet_normalize(pls.predict(fte)); m=evaluate(Y_test,Yp,LABEL_COLS)
            print_metrics(m,"M3 Test(ck)",2); return Yp,m
    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    Xtr=np.maximum(Xtr-Xtr.min(axis=1,keepdims=True),0)
    Xte=np.maximum(Xte-Xte.min(axis=1,keepdims=True),0)
    K=min(CFG["n_components_nmf"],len(X_train)-1)
    log.info(f"  NMF K={K}...")
    nmf=NMF(n_components=K,init="nndsvda",max_iter=3000,random_state=42)
    Ctr=nmf.fit_transform(Xtr); Cte=nmf.transform(Xte)
    Rtr=_band_int(Xtr,wavenumbers); Rte=_band_int(Xte,wavenumbers)
    ftr=np.hstack([Ctr,Rtr]); fte=np.hstack([Cte,Rte])
    sc=StandardScaler(); ftr=sc.fit_transform(ftr); fte=sc.transform(fte)
    log.info("  LOO-CV chọn n_lv PLSR...")
    best_lv,best_r2=2,-np.inf; loo=LeaveOneOut()
    max_lv=min(CFG["n_lv_plsr"],len(ftr)-1,ftr.shape[1])
    r2_vals=[]
    for lv in range(2,max_lv+1,2):
        pls_cv=PLSRegression(n_components=lv,max_iter=2000)
        Yp_cv=dirichlet_normalize(cross_val_predict(pls_cv,ftr,Y_train,cv=loo))
        rv=r2_score(Y_train,Yp_cv,multioutput="uniform_average")
        r2_vals.append(rv)
        if rv>best_r2: best_r2,best_lv=rv,lv
    plot_r2_curve(r2_vals,3,"MCR-ALS+PLSR (LOO-CV R² vs n_LV)")
    log.info(f"  Best LV={best_lv}, LOO R²={best_r2:.3f}")
    pls=PLSRegression(n_components=best_lv,max_iter=2000); pls.fit(ftr,Y_train)
    Yp_tr=dirichlet_normalize(pls.predict(ftr)); Yp_te=dirichlet_normalize(pls.predict(fte))
    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    log.info(f"  Time: {time.time()-t0:.1f}s")
    print_metrics(mtr,"M3 Train",2); print_metrics(mte,"M3 Test",2)
    save_checkpoint(3,{"nmf":nmf,"pls":pls,"sc":sc,"best_lv":best_lv,"wavenumbers":wavenumbers})
    save_results(3,mte,{"Y_pred":Yp_te,"Y_true":Y_test},
                 {"name":"MCR-ALS+PLSR","elapsed_s":time.time()-t0,
                  "best_lv":best_lv,"loo_r2":best_r2,"train_metrics":mtr})
    plot_predictions(Y_test,Yp_te,3,"MCR-ALS+PLSR",LABEL_COLS)
    if X_test_raw is not None:
        Xte_r=preprocess_batch(X_test_raw,"full")
        Xte_r=np.maximum(Xte_r-Xte_r.min(axis=1,keepdims=True),0)
        fte_r=sc.transform(np.hstack([nmf.transform(Xte_r),_band_int(Xte_r,wavenumbers)]))
        Yp_r=dirichlet_normalize(pls.predict(fte_r))
        plot_predictions_raw(Y_test_raw,Yp_r,sid_test_raw,3,"MCR-ALS+PLSR",LABEL_COLS)
    return Yp_te,mte
