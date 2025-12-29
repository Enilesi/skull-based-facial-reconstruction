import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from skull3d import Skull2Dto3D, freeze_encoder, SkullDataset
from skull3d.render import make_renderer, sample_cameras
from skull3d.losses import silhouette_bce, symmetry_chamfer, smoothness_losses, pseudo_mesh_chamfer
from skull3d.utils import ensure_dir, set_seed, device

def load_pseudo_mesh(obj_path: str, dev: torch.device):
    if obj_path is None or not os.path.isfile(obj_path):
        return None
    verts, faces, _ = load_obj(obj_path, device=dev)
    f = faces.verts_idx
    m = Meshes(verts=[verts], faces=[f])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use_pseudo", action="store_true")
    ap.add_argument("--w_sil", type=float, default=1.0)
    ap.add_argument("--w_sym", type=float, default=0.2)
    ap.add_argument("--w_smooth", type=float, default=0.2)
    ap.add_argument("--w_pseudo", type=float, default=0.5)
    ap.add_argument("--freeze_enc", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device()
    ensure_dir(args.out_dir)

    ds = SkullDataset(
        root=args.data_root,
        size=args.image_size,
        use_pseudo_meshes=args.use_pseudo,
        augment=True,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = Skull2Dto3D(image_size=args.image_size).to(dev)
    freeze_encoder(model, freeze=args.freeze_enc)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    renderer = make_renderer(args.image_size, dev)

    step = 0
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{args.epochs}")
        for batch in pbar:
            img = batch["img"].to(dev)
            mask = batch["mask"].to(dev)
            sex_id = batch["sex_id"].to(dev)

            meshes = model(img, sex_id)

            cams = sample_cameras(batch_size=img.shape[0], device=dev)
            sil = renderer(meshes_world=meshes, cameras=cams)[..., 3]
            sil = sil[:, None, ...]
            loss_sil = silhouette_bce(sil, mask)

            loss_sym = symmetry_chamfer(meshes, n_points=2048)
            loss_smooth = smoothness_losses(meshes, w_lap=1.0, w_norm=0.2)

            loss_pseudo = torch.tensor(0.0, device=dev)
            if args.use_pseudo:
                pseudo_meshes = []
                for p in batch["pseudo"]:
                    pm = load_pseudo_mesh(p, dev)
                    pseudo_meshes.append(pm)
                ok = [m is not None for m in pseudo_meshes]
                if any(ok):
                    pred_list = []
                    pseudo_list = []
                    for i, m in enumerate(pseudo_meshes):
                        if m is None:
                            continue
                        pred_list.append(meshes[i:i+1])
                        pseudo_list.append(m)
                    pred_cat = pred_list[0]
                    pseudo_cat = pseudo_list[0]
                    for i in range(1, len(pred_list)):
                        pred_cat = pred_cat.join_batch(pred_list[i])
                        pseudo_cat = pseudo_cat.join_batch(pseudo_list[i])
                    loss_pseudo = pseudo_mesh_chamfer(pred_cat, pseudo_cat, n_points=4096)

            loss = (
                args.w_sil * loss_sil
                + args.w_sym * loss_sym
                + args.w_smooth * loss_smooth
                + args.w_pseudo * loss_pseudo
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            pbar.set_postfix({
                "loss": float(loss.detach().cpu()),
                "sil": float(loss_sil.detach().cpu()),
                "sym": float(loss_sym.detach().cpu()),
                "sm": float(loss_smooth.detach().cpu()),
                "ps": float(loss_pseudo.detach().cpu()),
            })

        ckpt = {
            "model": model.state_dict(),
            "epoch": ep,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if (ep + 1) % 50 == 0:
            torch.save(ckpt, os.path.join(args.out_dir, f"ep_{ep+1}.pt"))

if __name__ == "__main__":
    main()
