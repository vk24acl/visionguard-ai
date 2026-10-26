import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

def splits(y):
    train,rest=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=42)
    cal,rest=train_test_split(rest,test_size=.625,stratify=y[rest],random_state=43)
    val,test=train_test_split(rest,test_size=.6,stratify=y[rest],random_state=44)
    return train,cal,val,test

def neural_logits(x,weights,biases):
    h=np.asarray(x,dtype=float)
    for i,(w,b) in enumerate(zip(weights,biases)):
        h=h@w+b
        if i<len(weights)-1:h=np.maximum(0,h)
    return h

def temperature_fit(logits,y):
    result=minimize_scalar(lambda t:log_loss(y,softmax(logits/t,axis=1),labels=np.arange(10)),bounds=(.2,5),method='bounded')
    return float(result.x)

def select_threshold(y,p,target=.98,min_samples=30):
    confidence=p.max(axis=1);correct=p.argmax(axis=1)==y
    valid=[]
    for threshold in np.unique(confidence):
        mask=confidence>=threshold
        if mask.sum()>=min_samples and correct[mask].mean()>=target:valid.append(float(threshold))
    return min(valid) if valid else 1.000001

def metrics(y,p,threshold):
    accept=p.max(axis=1)>=threshold
    return {'accuracy':float(accuracy_score(y,p.argmax(axis=1))),'nll':float(log_loss(y,p,labels=np.arange(10))),'coverage':float(accept.mean()),'accepted_count':int(accept.sum()),'selective_accuracy':float((p.argmax(axis=1)[accept]==y[accept]).mean()) if accept.any() else None}

def corrupt(x,kind,seed=9):
    a=np.asarray(x).copy();r=np.random.default_rng(seed)
    if kind=='noise':a=np.clip(a+r.normal(0,.25,a.shape),0,1)
    elif kind=='occlusion':a.reshape(-1,8,8)[:,3:5,:]=0
    elif kind=='shift':
        images=a.reshape(-1,8,8).copy();a.reshape(-1,8,8)[:,:,1:]=images[:,:,:-1];a.reshape(-1,8,8)[:,:,0]=0
    elif kind!='clean':raise ValueError('unknown corruption')
    return a

def run(output='reports'):
    ds=load_digits();x=ds.data/16;y=ds.target
    train,cal,val,test=splits(y)
    model=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=350,early_stopping=True,n_iter_no_change=20,random_state=42).fit(x[train],y[train])
    baseline=LogisticRegression(max_iter=1000).fit(x[train],y[train])
    logits=lambda a:neural_logits(a,model.coefs_,model.intercepts_)
    temperature=temperature_fit(logits(x[cal]),y[cal])
    pval=softmax(logits(x[val])/temperature,axis=1)
    threshold=select_threshold(y[val],pval)
    ptest=softmax(logits(x[test])/temperature,axis=1)
    result={'seed':42,'dataset':'scikit-learn load_digits (UCI optical recognition), 1797 images, 8x8 pixels','architecture':[64,64,32,10],'split_sizes':dict(zip(['train','calibration','validation','test'],map(len,[train,cal,val,test]))),'temperature':temperature,'validation_threshold':threshold,'validation':metrics(y[val],pval,threshold),'baseline_test':metrics(y[test],baseline.predict_proba(x[test]),0),'uncalibrated_test':metrics(y[test],model.predict_proba(x[test]),0),'conditions':{},'confusion_matrix':confusion_matrix(y[test],ptest.argmax(axis=1)).tolist(),'risk_coverage':[],'sample_predictions':[]}
    for condition in ['clean','noise','occlusion','shift']:
        result['conditions'][condition]=metrics(y[test],softmax(logits(corrupt(x[test],condition))/temperature,axis=1),threshold)
    for cutoff in [0,.5,.7,.8,.9,.95,.99]:result['risk_coverage'].append(dict(threshold=cutoff,**metrics(y[test],ptest,cutoff)))
    for i in range(min(40,len(test))):
        result['sample_predictions'].append({'pixels':x[test[i]].tolist(),'label':int(y[test[i]]),'prediction':int(ptest[i].argmax()),'confidence':float(ptest[i].max())})
    export={'weights':[w.tolist() for w in model.coefs_],'biases':[b.tolist() for b in model.intercepts_],'temperature':temperature,'threshold':threshold,'input_scale':'64 pixels in [0,1], row-major 8x8'}
    # Export parity is a release check, using an independent forward pass.
    np.testing.assert_allclose(softmax(logits(x[test]),axis=1),model.predict_proba(x[test]),atol=1e-10)
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    for filename,data in [('metrics.json',result),('model.json',export)]: (path/filename).write_text(json.dumps(data,indent=2)+'\n')
    m=result['conditions']['clean']
    (path/'summary.md').write_text(f"# Measured results\n\nHeld-out accuracy **{m['accuracy']:.1%}**, accepted coverage **{m['coverage']:.1%}**.\n\nTemperature {temperature:.3f}; confidence threshold {threshold:.3f}, selected on validation only.\n\nNoise, occlusion and shift results are in metrics.json. A validation accuracy target is not a guarantee on test data or unfamiliar images.\n")
    return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
