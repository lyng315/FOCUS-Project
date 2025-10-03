# n4/adaptive_compress.py
import argparse, torch, numpy as np, json
from sklearn.cluster import MiniBatchKMeans

def kavtc_select(feats, alpha=0.2, min_k=50, score_mode='dot'):
    N = feats.shape[0]
    k = max(min_k, int(round(alpha * N)))
    if score_mode == 'norm':
        scores = np.linalg.norm(feats, axis=1)
    else:
        centroid = feats.mean(axis=0)
        scores = feats.dot(centroid)
    idx = np.argsort(scores)[::-1][:k]
    return sorted(list(idx))

def svtc_grid_pool(feats, coords, block_size=1024):
    coords = np.array(coords)
    grid_x = (coords[:,0] // block_size).astype(int)
    grid_y = (coords[:,1] // block_size).astype(int)
    keep=set()
    for gx in np.unique(grid_x):
        for gy in np.unique(grid_y):
            ids = np.where((grid_x==gx)&(grid_y==gy))[0]
            if len(ids)==0: continue
            if len(ids)==1:
                keep.add(int(ids[0])); continue
            n_clusters = max(1, min(10, len(ids)//10))
            if n_clusters==1:
                c = feats[ids].mean(axis=0)
                d = ((feats[ids]-c)**2).sum(axis=1)
                keep.add(int(ids[np.argmin(d)]))
            else:
                km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
                lab = km.fit_predict(feats[ids])
                for lc in np.unique(lab):
                    cid = ids[lab==lc]
                    c = feats[cid].mean(axis=0)
                    d = ((feats[cid]-c)**2).sum(axis=1)
                    keep.add(int(cid[np.argmin(d)]))
    return sorted(list(keep))

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True)
    p.add_argument("--mode", choices=["kavtc","svtc"], default="kavtc")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--out_json", default=None)
    args=p.parse_args()

    obj = torch.load(args.pt, map_location='cpu')
    feats = obj['features'].cpu().numpy()
    coords = obj.get('coords', None)
    if args.mode=="kavtc":
        sel = kavtc_select(feats, alpha=args.alpha)
    else:
        if coords is None:
            raise RuntimeError("coords required for svtc")
        sel = svtc_grid_pool(feats, coords, block_size=args.block_size)
    if args.out_json:
        import json, os
        os.makedirs("compress", exist_ok=True)
        sel_python = [int(x) for x in sel]
        with open(args.out_json, "w") as f:
            json.dump({"selected_idx": sel_python}, f)
    print("Selected", len(sel), "of", feats.shape[0])
