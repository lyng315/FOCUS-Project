# n4/create_bag_h5.py
import argparse, torch, os, h5py, numpy as np, json

def make_bag(pt_path, sel_idx, out_path, label=None):
    obj = torch.load(pt_path, map_location='cpu')
    feats = obj['features'].cpu().numpy().astype('float32')
    patch_ids = obj.get('patch_ids', [str(i) for i in range(len(feats))])
    coords = obj.get('coords', None)
    sel = sel_idx
    sel_feats = feats[sel]
    sel_ids = [patch_ids[i] for i in sel]
    sel_coords = (np.array(coords)[sel] if coords is not None else np.zeros((len(sel),2), dtype='int32'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path,'w') as f:
        f.create_dataset('features', data=sel_feats)
        f.create_dataset('patch_ids', data=np.array(sel_ids, dtype='S'))
        f.create_dataset('coords', data=np.array(sel_coords, dtype='int32'))
        if label is not None:
            f.attrs['label'] = str(label)
        f.attrs['n_original'] = int(len(feats))
        f.attrs['n_selected'] = int(len(sel_feats))
    print("Saved bag", out_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True)
    parser.add_argument("--selected_json", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", required=True)
    args=parser.parse_args()
    sel = json.load(open(args.selected_json)) 
    # supports key 'selected_idx' or 'keep_idx'
    if 'selected_idx' in sel:
        idxs=sel['selected_idx']
    elif 'keep_idx' in sel:
        idxs=sel['keep_idx']
    else:
        raise RuntimeError("selected idx not found")
    make_bag(args.pt, idxs, args.out, label=args.label)
