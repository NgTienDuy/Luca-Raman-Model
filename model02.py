"""model02.py — PCA + Ridge Regression"""
import numpy as np, time
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from utils import (preprocess_batch, evaluate, print_metrics, save_checkpoint,
                   load_checkpoint, save_results, plot_predictions,
                   plot_predictions_raw, plot_r2_curve, get_logger)
from config import MODEL_CONFIGS, LABEL_COLS
log=get_logger("model02"); CFG=MODEL_CONFIGS[2]

def run(X_train,X_test,Y_train,Y_test,retrain=False,
        X_test_raw=None,Y_test_raw=None,sid_test_raw=None,**kw):
    log.info("="*60); log.info("Model 2 — PCA + Ridge")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    t0=time.time()
    if not retrain:
        ck=load_checkpoint(2)
        if ck:
            sc,pca,ridge=ck["sc"],ck["pca"],ck["ridge"]
            Xte=pca.transform(sc.transform(preprocess_batch(X_test,"full")))
            Yp=softmax(ridge.predict(Xte),axis=1); m=evaluate(Y_test,Yp,LABEL_COLS)
            print_metrics(m,"M2 Test(ck)",2); return Yp,m
    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    n_comp=min(CFG["n_components"],len(X_train)-1,X_train.shape[1])
    pca=PCA(n_components=n_comp); Xtr_p=pca.fit_transform(Xtr); Xte_p=pca.transform(Xte)
    log.info(f"  PCA: {pca.n_components_} components (var={pca.explained_variance_ratio_.sum():.3f})")
    eps=1e-6; Ylog=np.log(np.clip(Y_train,eps,1.)); Ylog-=Ylog.mean(axis=1,keepdims=True)
    # R² curve over n_components
    r2_vals=[]
    for nc in range(2, n_comp+1, max(1,n_comp//10)):
        pca_c=PCA(n_components=nc); Xtr_c=pca_c.fit_transform(Xtr)
        Xte_c=pca_c.transform(Xte)
        ridge_c=Ridge(alpha=CFG["alpha"]); ridge_c.fit(Xtr_c,Ylog)
        r2_vals.append(evaluate(Y_test,softmax(ridge_c.predict(Xte_c),axis=1),LABEL_COLS)["r2"])
    plot_r2_curve(r2_vals,2,"PCA+Ridge (n_components sweep)")
    ridge=Ridge(alpha=CFG["alpha"]); ridge.fit(Xtr_p,Ylog)
    Yp_tr=softmax(ridge.predict(Xtr_p),axis=1); Yp_te=softmax(ridge.predict(Xte_p),axis=1)
    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    log.info(f"  Time: {time.time()-t0:.1f}s")
    print_metrics(mtr,"M2 Train",2); print_metrics(mte,"M2 Test",2)
    save_checkpoint(2,{"sc":sc,"pca":pca,"ridge":ridge})
    save_results(2,mte,{"Y_pred":Yp_te,"Y_true":Y_test},
                 {"name":"PCA + Ridge","elapsed_s":time.time()-t0,"train_metrics":mtr})
    plot_predictions(Y_test,Yp_te,2,"PCA+Ridge",LABEL_COLS)
    if X_test_raw is not None:
        Xte_r=pca.transform(sc.transform(preprocess_batch(X_test_raw,"full")))
        Yp_r=softmax(ridge.predict(Xte_r),axis=1)
        plot_predictions_raw(Y_test_raw,Yp_r,sid_test_raw,2,"PCA+Ridge",LABEL_COLS)
    return Yp_te,mte
