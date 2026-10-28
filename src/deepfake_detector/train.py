"""Fine-tune EfficientNet-B0 from a face-image manifest."""
import argparse, json, logging, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .dataset import FaceManifestDataset
from .model import build_model

log=logging.getLogger(__name__)
def metrics(labels,scores):
    pred=[int(x>=.5) for x in scores]
    out={"accuracy":accuracy_score(labels,pred),"precision":precision_score(labels,pred,zero_division=0),
         "recall":recall_score(labels,pred,zero_division=0),"f1":f1_score(labels,pred,zero_division=0)}
    out["roc_auc"]=roc_auc_score(labels,scores) if len(set(labels))==2 else None
    return out
def run_training(manifest, output, epochs=5, batch_size=8, lr=1e-4, weight_decay=1e-4, seed=42, freeze_features=False, num_workers=0, resume=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds=FaceManifestDataset(manifest,"train",True); val_ds=FaceManifestDataset(manifest,"val",False)
    if not len(train_ds) or not len(val_ds): raise ValueError("Training and validation splits must both contain images")
    tr=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=device.type=="cuda")
    va=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=device.type=="cuda")
    model=build_model(pretrained=resume is None).to(device)
    if freeze_features:
        for p in model.features.parameters(): p.requires_grad=False
    if resume:
        ck=torch.load(resume,map_location=device,weights_only=True); model.load_state_dict(ck["state_dict"])
    counts=np.bincount([row["_label"] for row in train_ds.rows],minlength=2)
    weights=torch.tensor(len(train_ds)/(2*np.maximum(counts,1)),dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); best=-1; history=[]
    for epoch in range(epochs):
        model.train(); loss_sum=0
        for x,y,_ in tr:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                logits=model(x); loss=criterion(logits,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); loss_sum+=float(loss.item())*len(y)
        model.eval(); labels=[]; scores=[]; val_loss=0
        with torch.inference_mode():
            for x,y,_ in va:
                x=x.to(device); y=y.to(device); logits=model(x); val_loss+=float(criterion(logits,y))*len(y)
                scores.extend(torch.softmax(logits,1)[:,1].cpu().tolist()); labels.extend(y.cpu().tolist())
        row={"epoch":epoch+1,"train_loss":loss_sum/len(train_ds),"val_loss":val_loss/len(val_ds),**metrics(labels,scores)}; history.append(row); log.info("%s",row)
        if row["f1"]>best:
            best=row["f1"]; Path(output).parent.mkdir(parents=True,exist_ok=True)
            ck={"state_dict":model.state_dict(),"metadata":{"architecture":"efficientnet_b0","class_to_index":{"real":0,"fake":1},"image_size":224,"preprocessing":{"mean":[.485,.456,.406],"std":[.229,.224,.225]},"training_config":{"epochs":epochs,"batch_size":batch_size,"learning_rate":lr,"weight_decay":weight_decay,"seed":seed},"model_version":"0.1.0","dataset_manifest":str(Path(manifest).resolve())}}
            torch.save(ck,output)
    Path(str(output)+".history.json").write_text(json.dumps(history,indent=2),encoding="utf8")
    return history
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output",default="models/efficientnet_b0.pt"); p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--seed",type=int,default=42); p.add_argument("--freeze-features",action="store_true"); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--resume")
    a=p.parse_args(); logging.basicConfig(level=logging.INFO); run_training(a.manifest,a.output,a.epochs,a.batch_size,a.lr,a.weight_decay,a.seed,a.freeze_features,a.num_workers,a.resume)
if __name__=="__main__": main()
