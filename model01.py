"""model01.py — Softmax Regression"""
import numpy as np, time, os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from utils import (preprocess_batch, evaluate, print_metrics, save_checkpoint,
                   load_checkpoint, save_results, plot_predictions,
                   plot_predictions_raw, plot_r2_curve, get_logger)
from config import MODEL_CONFIGS, LABEL_COLS
log=get_logger("model01"); CFG=MODEL_CONFIGS[1]

def run(X_train,X_test,Y_train,Y_test,retrain=False,
        X_test_raw=None,Y_test_raw=None,sid_test_raw=None,**kw):
    log.info("="*60); log.info("Model 1 — Softmax Regression")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    t0=time.time()
    if not retrain:
        ck=load_checkpoint(1)
        if ck:
            log.info("  Tải checkpoint")
            sc,W=ck["sc"],ck["W"]
            Xte=sc.transform(preprocess_batch(X_test,"full"))
            Yp=softmax(Xte@W,axis=1); m=evaluate(Y_test,Yp,LABEL_COLS)
            print_metrics(m,"M1 Test(ck)",2); return Yp,m
    Xtr=preprocess_batch(X_train,"full"); Xte=preprocess_batch(X_test,"full")
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    eps=1e-6; Ylog=np.log(np.clip(Y_train,eps,1.)); Ylog-=Ylog.mean(axis=1,keepdims=True)
    # Sweep alpha to get R² curve
    alphas=[10,50,100,500,1000,5000,CFG["alpha"],50000]
    r2_vals=[]
    for a in alphas:
        ridge=Ridge(alpha=a,solver=CFG["solver"],max_iter=CFG["max_iter"])
        ridge.fit(Xtr,Ylog)
        Yp_a=softmax(Xte@ridge.coef_.T,axis=1)
        r2_vals.append(evaluate(Y_test,Yp_a,LABEL_COLS)["r2"])
    plot_r2_curve(r2_vals,1,"Softmax Regression (alpha sweep)")
    ridge=Ridge(alpha=CFG["alpha"],solver=CFG["solver"],max_iter=CFG["max_iter"])
    ridge.fit(Xtr,Ylog); W=ridge.coef_.T
    Yp_tr=softmax(Xtr@W,axis=1); Yp_te=softmax(Xte@W,axis=1)
    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    log.info(f"  Time: {time.time()-t0:.1f}s")
    print_metrics(mtr,"M1 Train",2); print_metrics(mte,"M1 Test",2)
    save_checkpoint(1,{"W":W,"sc":sc})
    save_results(1,mte,{"Y_pred":Yp_te,"Y_true":Y_test},
                 {"name":"Softmax Regression","elapsed_s":time.time()-t0,"train_metrics":mtr})
    plot_predictions(Y_test,Yp_te,1,"Softmax Regression",LABEL_COLS)
    if X_test_raw is not None:
        Xte_r=sc.transform(preprocess_batch(X_test_raw,"full"))
        Yp_r=softmax(Xte_r@W,axis=1)
        plot_predictions_raw(Y_test_raw,Yp_r,sid_test_raw,1,"Softmax Regression",LABEL_COLS)
    return Yp_te,mte
