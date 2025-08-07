import argparse as ap
import os
import yaml
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from ...emb.nn.model import StateEmbeddingModel
from ...emb.utils import compute_gene_overlap_cross_pert, get_embedding_cfg, get_precision_config
from ...emb.data import create_dataloader
from ...emb.inference import Inference


def add_arguments_eval(parser: ap.ArgumentParser):
    """Add arguments for embedding evaluation CLI."""
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--adata", required=True,
        help="Path to AnnData file"
    )
    parser.add_argument(
        "--config", required=False,
        help="Path to configuration file (optional)"
    )
    parser.add_argument(
        "--pert-col", default="gene",
        help="Column name for perturbation labels (default: gene)"
    )
    parser.add_argument(
        "--control-pert", default="non-targeting",
        help="Control perturbation label (default: non-targeting)"
    )
    parser.add_argument(
        "--gene-column", default="gene_name",
        help="Column name for gene names (default: gene_name)"
    )


def load_config(config_path: str | None = None):
    """Load config from YAML file or create default config."""
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        return cfg
    else:
        # Create minimal default config for inference
        cfg_dict = {
            "model": {
                "batch_size": 32,
                "rda": False
            },
            "dataset": {
                "P": 1000,
                "N": 1000
            },
            "validations": {
                "diff_exp": {
                    "top_k_rank": 50,
                    "method": "wilcoxon"
                }
            },
            "embeddings": {
                "current": "default",
                "default": {
                    "all_embeddings": "path/to/embeddings.pt",
                    "size": 5120
                }
            }
        }
        return OmegaConf.create(cfg_dict)


def run_emb_eval(args):
    """
    Run embedding evaluation.
    """
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Loading AnnData: {args.adata}")
    print(f"Perturbation column: {args.pert_col}")
    print(f"Control perturbation: {args.control_pert}")
    
    if args.config:
        print(f"Loading config: {args.config}")
    else:
        print("Using default config")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Load AnnData
    adata = sc.read_h5ad(args.adata)
    print(f"Loaded AnnData with shape: {adata.shape}")
    
    # Create inference object and load model
    print("Creating inference object and loading model...")
    
    # Try to load protein embeddings if config is available
    protein_embeds = None
    if args.config and os.path.exists(args.config):
        try:
            embedding_file = os.path.join(os.path.dirname(args.checkpoint), "protein_embeddings.pt")
            if os.path.exists(embedding_file):
                protein_embeds = torch.load(embedding_file, weights_only=False, map_location="cpu")
                print(f"Loaded protein embeddings from: {embedding_file}")
        except Exception as e:
            print(f"Warning: Could not load protein embeddings: {e}")
    
    inference = Inference(cfg=cfg, protein_embeds=protein_embeds)
    inference.load_model(args.checkpoint)
    adata = inference._convert_to_csr(adata)

    model = inference.model
    
    # Type assertion to help linter understand model is not None
    assert model is not None, "Model failed to load from checkpoint"
    
    print("Model loaded successfully")
    
    # Create dataloader
    print("Creating dataloader...")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    precision = get_precision_config(device_type=device_type)
    dataloader = create_dataloader(
        cfg,
        adata=adata,
        adata_name="eval_dataset",
        shuffle=False,
        gene_column=args.gene_column,
        precision=precision,
    )
    
    # Compute embeddings and predictions
    print("Computing embeddings and predictions...")
    emb_batches = []
    ds_emb_batches = []
    logprob_batches = []
    
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=precision):
            for batch in tqdm(dataloader, desc="Processing batches"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # Compute embeddings
                _, _, _, emb, ds_emb = model._compute_embedding_for_batch(batch)
                
                # Get gene embeddings
                try:
                    gene_embeds = model.get_gene_embedding(adata.var.index)
                except:
                    gene_embeds = model.get_gene_embedding(adata.var["gene_symbols"])
                
                # Handle dataset embeddings
                if hasattr(model, 'dataset_token') and model.dataset_token is not None:
                    ds_emb = model.dataset_embedder(ds_emb)
                
                # Store embeddings
                if emb is not None:
                    emb_batches.append(emb.detach().cpu().float().numpy())
                if ds_emb is not None:
                    ds_emb_batches.append(ds_emb.detach().cpu().float().numpy())
                
                # Resize batch and decode
                task_counts = None
                Y = batch[2]
                nan_y = Y.masked_fill(Y == 0, float("nan"))[:, : cfg.dataset.P + cfg.dataset.N]
                task_counts = torch.nanmean(nan_y, dim=1) if cfg.model.rda else None
                
                # Ensure task_counts is on the same device as other tensors
                if task_counts is not None:
                    task_counts = task_counts.to(model.device)

                merged_embs = model.__class__.resize_batch(emb, gene_embeds, task_counts=task_counts, ds_emb=ds_emb)
                logprobs_batch = model.binary_decoder(merged_embs)
                logprobs_batch = logprobs_batch.detach().cpu().float().numpy()
                logprob_batches.append(logprobs_batch.squeeze())
    
    # Combine batches
    logprob_batches = np.vstack(logprob_batches)
    emb_combined = np.vstack(emb_batches)
    ds_emb_combined = np.vstack(ds_emb_batches)
    adata.obsm["X_emb"] = np.concatenate([emb_combined, ds_emb_combined], axis=-1)
    
    # Create predictions DataFrame
    probs_df = pd.DataFrame(logprob_batches)
    probs_df[args.pert_col] = adata.obs[args.pert_col].values
    
    # Get top-k genes for each perturbation
    k = cfg.validations.diff_exp.top_k_rank
    probs_df = probs_df.groupby(args.pert_col).mean()
    ctrl = probs_df.loc[args.control_pert].values
    pert_effects = np.abs(probs_df - ctrl)
    top_k_indices = np.argsort(pert_effects.values, axis=1)[:, -k:][:, ::-1]
    top_k_genes = np.array(adata.var.index)[top_k_indices]
    pred_de_genes = pd.DataFrame(top_k_genes)
    pred_de_genes.index = pert_effects.index.values
    
    print(f"Predicted DEGs shape: {pred_de_genes.shape}")
    
    # Compute ground truth DEGs
    print("Computing ground truth DEGs...")
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(
        adata,
        groupby=args.pert_col,
        reference=args.control_pert,
        rankby_abs=True,
        n_genes=k,
        method=cfg.validations.diff_exp.method,
        use_raw=False,
    )
    true_de_genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
    true_de_genes = true_de_genes.T
    
    print(f"Ground truth DEGs shape: {true_de_genes.shape}")
    
    # Compute overlap metrics
    print("Computing gene overlap metrics...")
    de_metrics = compute_gene_overlap_cross_pert(
        pred_de_genes, true_de_genes, control_pert=args.control_pert, k=k
    )
    
    # Print results
    mean_overlap = np.array(list(de_metrics.values())).mean()
    print(f"\nResults:")
    print(f"Mean gene overlap: {mean_overlap:.4f}")
    print(f"Number of perturbations evaluated: {len(de_metrics)}")
    
    return de_metrics, mean_overlap


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    add_arguments_eval(parser)
    args = parser.parse_args()
    run_emb_eval(args)