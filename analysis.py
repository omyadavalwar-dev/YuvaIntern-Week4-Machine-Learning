"""YuvaIntern Week 4: species classification with training-only model selection.
Extract the complete package, install requirements, and run python analysis.py.
"""
# %% Setup and reproducible data preparation
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
import sklearn, scipy
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('error', category=ConvergenceWarning)
ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for name in ['data','tables','charts']:
    (ROOT/name).mkdir(exist_ok=True)
features=['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
classes=['Adelie','Chinstrap','Gentoo']
raw=pd.read_csv(ROOT/'data/penguins_raw.csv',na_values=['NA'])
df=raw.copy()
for col in ['species','island','sex']:
    df[col]=df[col].astype('string').str.strip()
for col in features:
    df[col]=pd.to_numeric(df[col],errors='raise')
assert df.duplicated().sum()==0
df.insert(0,'source_row',np.arange(1,len(df)+1))
excluded=df.loc[df[features].isna().any(axis=1)].copy()
df=df.dropna(subset=features+['species']).copy()
assert len(df)==342 and len(excluded)==2
assert df[features].gt(0).all().all() and set(df.species)==set(classes)
df.to_csv(ROOT/'data/penguins_model_data.csv',index=False)
excluded.to_csv(ROOT/'data/excluded_rows.csv',index=False)

# %% Split before fitting any transformation; retain the same split for all models
train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.20,
                                    random_state=42,stratify=df.species)
assert not set(train_idx)&set(test_idx)
X_train=df.iloc[train_idx][features]; y_train=df.iloc[train_idx].species
X_test=df.iloc[test_idx][features]; y_test=df.iloc[test_idx].species
split=pd.DataFrame({'source_row':df.source_row,'species':df.species,'split':'train'})
split.loc[df.index[test_idx],'split']='test'
split.to_csv(ROOT/'tables/split_manifest.csv',index=False)
pd.crosstab(split.species,split['split']).reindex(classes).to_csv(ROOT/'tables/split_counts.csv')

# %% Fixed candidate comparison on training data only
# Highest mean five-fold CV macro F1 wins; insertion order breaks exact ties.
# C=1 and tree depth=3 were fixed before evaluation; there is no test-set tuning.
models={
 'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000,solver='lbfgs')),
 'Decision tree':DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,random_state=42),
 'Majority baseline':DummyClassifier(strategy='most_frequent')}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
rows=[]; fold_rows=[]
for name,model in models.items():
    scores=cross_validate(model,X_train,y_train,cv=cv,
        scoring={'accuracy':'accuracy','macro_f1':'f1_macro'},return_train_score=True)
    rows.append({'model':name,'train_macro_f1':scores['train_macro_f1'].mean(),
        'cv_macro_f1':scores['test_macro_f1'].mean(),
        'cv_macro_f1_sd':scores['test_macro_f1'].std(ddof=1),
        'cv_accuracy':scores['test_accuracy'].mean()})
    for fold in range(5):
        fold_rows.append({'model':name,'fold':fold+1,
            'train_macro_f1':scores['train_macro_f1'][fold],
            'validation_macro_f1':scores['test_macro_f1'][fold],
            'validation_accuracy':scores['test_accuracy'][fold]})
comparison=pd.DataFrame(rows).set_index('model')
comparison.to_csv(ROOT/'tables/cv_comparison.csv')
pd.DataFrame(fold_rows).to_csv(ROOT/'tables/cv_folds.csv',index=False)
winner=comparison.cv_macro_f1.idxmax()
selected=clone(models[winner]).fit(X_train,y_train)

# %% Final held-out evaluation of the selected model and fixed baseline
pred=selected.predict(X_test); prob=selected.predict_proba(X_test)
assert list(selected.classes_)==classes and np.allclose(prob.sum(axis=1),1)
baseline=clone(models['Majority baseline']).fit(X_train,y_train)
base_pred=baseline.predict(X_test)
acc=accuracy_score(y_test,pred); correct=int((pred==y_test.to_numpy()).sum()); n=len(y_test)
# Wilson interval is a conditional binomial summary, assuming independent test records.
z=1.959963984540054
center=(acc+z*z/(2*n))/(1+z*z/n)
half=z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
report=pd.DataFrame(classification_report(y_test,pred,labels=classes,output_dict=True,zero_division=0)).T
report.to_csv(ROOT/'tables/classification_report.csv')
cm=confusion_matrix(y_test,pred,labels=classes)
pd.DataFrame(cm,index=classes,columns=classes).to_csv(ROOT/'tables/confusion_matrix.csv')
predictions=df.iloc[test_idx][['source_row','species']+features].copy()
predictions['prediction']=pred
for i,label in enumerate(classes):predictions['prob_'+label]=prob[:,i]
predictions['correct']=pred==y_test.to_numpy()
predictions.to_csv(ROOT/'tables/test_predictions.csv',index=False)
predictions.loc[~predictions.correct].to_csv(ROOT/'tables/test_errors.csv',index=False)
result={'selected_model':winner,'n_raw':len(raw),'n_used':len(df),
 'n_train':len(train_idx),'n_test':n,'correct':correct,'errors':n-correct,
 'test_accuracy':acc,'test_error_rate':1-acc,
 'test_macro_f1':f1_score(y_test,pred,average='macro'),
 'test_balanced_accuracy':balanced_accuracy_score(y_test,pred),
 'test_macro_ovr_auc':roc_auc_score(y_test,prob,labels=classes,multi_class='ovr',average='macro'),
 'accuracy_wilson_95':[center-half,center+half],
 'baseline_test_accuracy':accuracy_score(y_test,base_pred),
 'baseline_test_macro_f1':f1_score(y_test,base_pred,average='macro'),
 'refit_train_accuracy':accuracy_score(y_train,selected.predict(X_train)),
 'source_sha256':hashlib.sha256((ROOT/'data/penguins_raw.csv').read_bytes()).hexdigest(),
 'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
             'scikit-learn':sklearn.__version__,'matplotlib':matplotlib.__version__}}
(ROOT/'tables/results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
print(comparison.to_string())

# %% Three evaluation charts
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def save(name):
    plt.tight_layout();plt.savefig(ROOT/'charts'/name,dpi=170,bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(8,3.9))
x=np.arange(len(comparison)); w=.34
ax.bar(x-w/2,comparison.train_macro_f1,w,label='Training folds',color='#759CAD')
ax.bar(x+w/2,comparison.cv_macro_f1,w,yerr=comparison.cv_macro_f1_sd,capsize=4,
       label='Validation folds (mean ± SD)',color='#237A70')
ax.set(xticks=x,xticklabels=['Logistic\nregression','Decision\ntree','Majority\nbaseline'],
       ylim=(0,1.08),ylabel='Macro F1',title='Model choice uses training data cross validation')
ax.legend(loc='lower left');save('01_model_comparison.png')
fig,ax=plt.subplots(figsize=(7,4))
im=ax.imshow(cm,cmap='Blues',vmin=0)
for i in range(3):
    for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,
       color='white' if cm[i,j]>cm.max()/2 else 'black')
ax.set(xticks=range(3),yticks=range(3),xticklabels=classes,yticklabels=classes,
       xlabel='Predicted species',ylabel='Actual species',title=f'{winner} test confusion matrix (n={n})')
fig.colorbar(im,ax=ax,label='Test records');save('02_confusion_matrix.png')
fig,ax=plt.subplots(figsize=(7,4))
aucs={}
for i,(label,color,style) in enumerate(zip(classes,['#236A93','#C26B27','#28826C'],['-','--',':'])):
    fpr,tpr,_=roc_curve((y_test==label).astype(int),prob[:,i]); a=auc(fpr,tpr);aucs[label]=float(a)
    ax.plot(fpr,tpr,label=f'{label} vs rest (AUC {a:.3f})',color=color,ls=style,lw=2)
ax.plot([0,1],[0,1],color='gray',ls='--',label='Chance reference')
ax.set(xlim=(-.02,1.02),ylim=(-.02,1.05),xlabel='False positive rate',ylabel='True positive rate',
       title='Test ROC curves use class probabilities')
ax.legend(loc='lower right',fontsize=9);save('03_roc_curves.png')
(ROOT/'tables/class_auc.json').write_text(json.dumps(aucs,indent=2))
