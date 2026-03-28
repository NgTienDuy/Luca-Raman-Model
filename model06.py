"""model06.py — Spectral Bond Analysis + XGBoost"""
import numpy as np, time, os
from utils import (preprocess_batch, evaluate, print_metrics, save_checkpoint,
                   load_checkpoint, save_results, plot_predictions,
                   plot_predictions_raw, plot_r2_curve, plot_bond_detect_batch,
                   get_logger, dirichlet_normalize)
from spectral_knowledge import get_active_regions, extract_bond_features, extract_derivative_features
from config import MODEL_CONFIGS, LABEL_COLS, get_model_dir
log=get_logger("model06"); CFG=MODEL_CONFIGS[6]

def _feats(X, wns):
    regions=get_active_regions(wns)
    bf=np.array([extract_bond_features(s,wns,regions) for s in X])
    df=np.array([extract_derivative_features(s,wns,n_seg=8) for s in X])
    return np.hstack([bf,df]).astype(np.float32), regions

def run(X_train,X_test,Y_train,Y_test,wavenumbers=None,retrain=False,
        X_test_raw=None,Y_test_raw=None,sid_test_raw=None,**kw):
    log.info("="*60); log.info("Model 6 — Bond + XGBoost")
    log.info(f"  N_train={len(X_train)}, N_test={len(X_test)}"); log.info("="*60)
    t0=time.time()
    try: import xgboost as xgb
    except ImportError: log.error("pip install xgboost"); return None,None
    if wavenumbers is None: wavenumbers=np.linspace(267,2004,X_train.shape[1])

    if not retrain:
        ck=load_checkpoint(6)
        if ck:
            models,fn=ck["models"],ck["feature_names"]
            Xte=preprocess_batch(X_test,"full")
            fte,_=_feats(Xte,wavenumbers); dtest=xgb.DMatrix(fte,feature_names=fn)
            Yp=dirichlet_normalize(np.column_stack([m.predict(dtest) for m in models]))
            mm=evaluate(Y_test,Yp,LABEL_COLS); print_metrics(mm,"M6 Test(ck)",2); return Yp,mm

    Xtr_proc=preprocess_batch(X_train,"full"); Xte_proc=preprocess_batch(X_test,"full")
    log.info("  Trích xuất features...")
    ftr,regions=_feats(Xtr_proc,wavenumbers); fte,_=_feats(Xte_proc,wavenumbers)
    log.info(f"  Feature shape: {ftr.shape}")
    fn=[]
    for br in regions:
        for feat in ["area","peak_h","peak_pos","FWHM","skew"]: fn.append(f"{br.name}_{feat}")
    for seg in range(8): fn+=[f"d1_s{seg}",f"d2_s{seg}"]
    fn=fn[:ftr.shape[1]]
    models=[]; imp=np.zeros((6,len(fn)))
    for j,aa in enumerate(LABEL_COLS):
        log.info(f"  [{j+1}/6] {aa}...")
        n_val=max(2,int(0.2*len(ftr))); idx=np.random.RandomState(j).permutation(len(ftr)); tr_i,va_i=idx[n_val:],idx[:n_val]
        dtrain=xgb.DMatrix(ftr[tr_i],label=Y_train[tr_i,j],feature_names=fn)
        dval=xgb.DMatrix(ftr[va_i],label=Y_train[va_i,j],feature_names=fn)
        params={"objective":"reg:squarederror","learning_rate":CFG["learning_rate"],
                "max_depth":CFG["max_depth"],"subsample":CFG["subsample"],
                "colsample_bytree":CFG["colsample_bytree"],"tree_method":"hist",
                "seed":42,"verbosity":0,"lambda":CFG["lambda"],"alpha":CFG["alpha"]}
        bst=xgb.train(params,dtrain,num_boost_round=CFG["n_estimators"],
                      evals=[(dval,"val")],early_stopping_rounds=CFG["early_stopping_rounds"],verbose_eval=False)
        models.append(bst)
        for fname,fscore in bst.get_score(importance_type="gain").items():
            if fname in fn: imp[j,fn.index(fname)]=fscore
    dtest=xgb.DMatrix(fte,feature_names=fn); dtr=xgb.DMatrix(ftr,feature_names=fn)
    Yp_te=dirichlet_normalize(np.column_stack([m.predict(dtest) for m in models]))
    Yp_tr=dirichlet_normalize(np.column_stack([m.predict(dtr)   for m in models]))
    mtr=evaluate(Y_train,Yp_tr,LABEL_COLS); mte=evaluate(Y_test,Yp_te,LABEL_COLS)
    elapsed=time.time()-t0; log.info(f"  Time: {elapsed:.1f}s")
    print_metrics(mtr,"M6 Train",2); print_metrics(mte,"M6 Test",2)

    # R² curve (importance scores)
    imp_m=imp.mean(0); top_idx=np.argsort(imp_m)[-15:][::-1]
    plot_r2_curve([imp_m[i] for i in top_idx],6,"Top-15 Feature Importance (XGBoost)")

    # Bond detection plots for test samples
    if X_test_raw is not None:
        Xte_raw_proc=preprocess_batch(X_test_raw,"full")
        # Get unique test sample spectra (median per sample)
        unique_test_sids=list(dict.fromkeys(sid_test_raw))
        for sid in unique_test_sids:
            mask=sid_test_raw==sid
            s_rep=np.median(Xte_raw_proc[mask],axis=0)
            idx_s=np.where(mask)[0][0]
            plot_bond_detect_batch(s_rep[None], wavenumbers, [sid], 6,
                                   "Bond+XGBoost", Y_test_raw[idx_s:idx_s+1], LABEL_COLS)
        # Raw scatter
        fte_r,_=_feats(Xte_raw_proc,wavenumbers)
        dtest_r=xgb.DMatrix(fte_r,feature_names=fn)
        Yp_r=dirichlet_normalize(np.column_stack([m.predict(dtest_r) for m in models]))
        plot_predictions_raw(Y_test_raw,Yp_r,sid_test_raw,6,"Bond+XGBoost",LABEL_COLS)

    plot_predictions(Y_test,Yp_te,6,"Bond+XGBoost",LABEL_COLS)
    save_checkpoint(6,{"models":models,"feature_names":fn,"importance":imp,"wavenumbers":wavenumbers})
    save_results(6,mte,{"Y_pred":Yp_te,"Y_true":Y_test},
                 {"name":"Bond+XGBoost","elapsed_s":elapsed,
                  "top5":[fn[i] for i in top_idx[:5] if i<len(fn)],
                  "train_metrics":mtr})

    # Chemistry
    try:
        from chemistry_report import batch_chemistry,mean_profile,format_report,plot_bond_contribution
        from dataclasses import asdict; import json
        md=get_model_dir(6)
        avg=mean_profile(batch_chemistry(Xtr_proc,wavenumbers,Y_train,LABEL_COLS))
        print(format_report(avg,"Model 6 — Bond+XGBoost",show_composition=True))
        plot_bond_contribution(avg,"M6-Bond+XGBoost",os.path.join(md,"model06_chemistry_bonds.png"))
        with open(os.path.join(md,"model06_chemistry.json"),"w") as ff:
            json.dump({"model":6,"profile":asdict(avg)},ff,indent=2,
                      default=lambda x:float(x) if hasattr(x,"__float__") else str(x))
    except Exception as ce: log.warning(f"  Chemistry: {ce}")
    return Yp_te,mte
