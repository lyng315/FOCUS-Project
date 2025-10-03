# n4/deduplicate.py
import argparse, torch, numpy as np
def dedup_with_faiss(feats, threshold):
    try:
        import faiss
    except:
        return None
    xb = feats.astype('float32')
    faiss.normalize_L2(xb)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    D,I = index.search(xb, 2)
    keep = []
    removed = set()
    for i in range(xb.shape[0]):
        if i in removed: continue
        sim = D[i,1]
        nei = int(I[i,1])
        if sim >= threshold:
            removed.add(nei)
        keep.append(i)
    keep = [i for i in keep if i not in removed]
    return sorted(keep)

def dedup_with_sklearn(feats, threshold):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    xb = normalize(feats)
    nbrs = NearestNeighbors(n_neighbors=2, metric='cosine').fit(xb)
    D,I = nbrs.kneighbors(xb)
    # D is cosine-distances; convert to similarity = 1 - dist
    keep=[]
    removed=set()
    for i in range(xb.shape[0]):
        if i in removed: continue
        sim = 1.0 - D[i,1]
        nei = int(I[i,1])
        if sim >= threshold:
            removed.add(nei)
        keep.append(i)
    keep=[i for i in keep if i not in removed]
    return sorted(keep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True)
    parser.add_argument("--sim_thresh", type=float, default=0.95)
    parser.add_argument("--out_json", default=None)
    args=parser.parse_args()

    obj = torch.load(args.pt, map_location='cpu')
    feats = obj['features'].cpu().numpy()
    keep = dedup_with_faiss(feats, args.sim_thresh)
    if keep is None:
        keep = dedup_with_sklearn(feats, args.sim_thresh)
    print("Kept:", len(keep), "of", feats.shape[0])
    if args.out_json:
        import json, os
        os.makedirs("compress", exist_ok=True)
        json.dump({"keep_idx": keep}, open(args.out_json, "w"))
