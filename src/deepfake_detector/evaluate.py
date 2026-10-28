"""Evaluate held-out videos by aggregating frame scores per video."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from torch.utils.data import DataLoader
from .dataset import FaceManifestDataset
from .model import load_checkpoint

def evaluate(checkpoint,manifest):
    ds=FaceManifestDataset(manifest,"test",False)
    if not len(ds): raise ValueError("Test split is empty")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model,_=load_checkpoint(checkpoint,device)
    groups=defaultdict(list); loader=DataLoader(ds,batch_size=16,num_workers=0)
    with torch.inference_mode():
        for x,y,ids in loader:
            probs=torch.softmax(model(x.to(device)),1)[:,1].cpu().tolist()
            for vid,label,s in zip(ids,y.tolist(),probs): groups[vid].append((label,s))
    labels=[]; scores=[]
    for vid,rows in groups.items():
        ys={r[0] for r in rows}
        if len(ys)!=1: raise ValueError(f"Conflicting labels for video {vid}")
        labels.append(next(iter(ys))); scores.append(float(__import__('numpy').median([r[1] for r in rows])))
    pred=[int(s>=.5) for s in scores]; tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {"video_count":len(labels),"accuracy":accuracy_score(labels,pred),"precision":precision_score(labels,pred,zero_division=0),"recall":recall_score(labels,pred,zero_division=0),"f1":f1_score(labels,pred,zero_division=0),"roc_auc":roc_auc_score(labels,scores) if len(set(labels))>1 else None,"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"false_positive_rate":fp/(fp+tn) if fp+tn else None}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--manifest",required=True); a=p.parse_args(); print(json.dumps(evaluate(a.checkpoint,a.manifest),indent=2))
if __name__=="__main__":main()
