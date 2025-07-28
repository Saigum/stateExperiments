import argparse
import os
import logging
import torch
import anndata
import h5py as h5 # Not directly used for loading in the main script now, but good to keep if used elsewhere
import numpy as np
import shutil

from pathlib import Path
from tqdm import tqdm
from torch import nn
from lightning.fabric import Fabric
from omegaconf import OmegaConf, DictConfig # Import OmegaConf

# Assuming these imports are correctly resolved based on your project structure
from state.emb.nn.model import StateEmbeddingModel
from state.emb.train.trainer import get_embeddings
from state.emb.data import create_dataloader
from state.emb.utils import get_embedding_cfg, get_precision_config
from typing import Optional
log = logging.getLogger(__name__)


# The Inference class itself doesn't need much change, as it already expects a 'cfg' object
# which will now be our OmegaConf DictConfig.
class Inference:
    def __init__(self, cfg: DictConfig, protein_embeds=None, fabric: Optional[Fabric] = None):
        if fabric is None:
            raise ValueError("A Fabric instance must be provided to Inference for multi-GPU support.")
        self.model = None
        self.collator = None
        self.protein_embeds = protein_embeds
        self._vci_conf = cfg # This will now be the OmegaConf object
        self.fabric = fabric

    def __load_dataset_meta(self, adata_path):
        with h5.File(adata_path, 'r') as h5f: # Ensure read mode
            attrs = dict(h5f["X"].attrs)
            if "encoding-type" in attrs:
                if attrs["encoding-type"] in ["csr_matrix", "csc_matrix"]:
                    num_cells = attrs["shape"][0] # type: ignore
                    num_genes = attrs["shape"][1]
                elif attrs["encoding-type"] == "array":
                    num_cells = h5f["X"].shape[0]
                    num_genes = h5f["X"].shape[1]
                else:
                    raise ValueError(f"Input file contains count mtx in unsupported encoding-type: {attrs['encoding-type']}")
            else:
                if hasattr(h5f["X"], "shape") and len(h5f["X"].shape) == 2:
                    num_cells = h5f["X"].shape[0]
                    num_genes = h5f["X"].shape[1]
                elif all(key in h5f["X"] for key in ["indptr", "indices", "data"]): # type: ignore
                    num_cells = len(h5f["X"]["indptr"]) - 1 # type: ignore
                    # This logic for num_genes might be flawed for empty sparse matrices
                    # A more robust way might be to peek at indices or rely on attrs if available
                    num_genes = attrs.get(
                        "shape", [0, h5f["X"]["indices"][:].max() + 1 if len(h5f["X"]["indices"]) > 0 else 0]
                    )[1]
                else:
                    raise ValueError("Cannot determine matrix format - no encoding-type and unrecognized structure")

        return {Path(adata_path).stem: (num_cells, num_genes)}

    def _save_data(self, input_adata_path, output_adata_path, obsm_key, data):
        """
        Save data in the output file. This function addresses following cases:
        - output_adata_path does not exist:
          In this case, the function copies the rest of the input file to the
          output file then adds the data to the output file.
        - output_adata_path exists but the dataset does not exist:
          In this case, the function adds the dataset to the output file.
        - output_adata_path exists and the dataset exists:
          In this case, the function resizes the dataset and appends the data to
          the dataset.
        """
        if not os.path.exists(output_adata_path):
            os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
            with h5.File(input_adata_path, 'r') as input_h5f: # Ensure read mode
                with h5.File(output_adata_path, "a") as output_h5f:
                    # Replicate the input data to the output file
                    for item_name in input_h5f.keys():
                        input_h5f.copy(item_name, output_h5f)
                    output_h5f.create_dataset(
                        f"/obsm/{obsm_key}", chunks=True, data=data, maxshape=(None, data.shape[1])
                    )
        else:
            with h5.File(output_adata_path, "a") as output_h5f:
                # If the dataset is added to an existing file that does not have the dataset
                if f"/obsm/{obsm_key}" not in output_h5f:
                    output_h5f.create_dataset(
                        f"/obsm/{obsm_key}", chunks=True, data=data, maxshape=(None, data.shape[1])
                    )
                else:
                    output_h5f[f"/obsm/{obsm_key}"].resize(
                        (output_h5f[f"/obsm/{obsm_key}"].shape[0] + data.shape[0]), axis=0
                    )
                    output_h5f[f"/obsm/{obsm_key}"][-data.shape[0] :] = data

    def load_model(self, checkpoint):
        if self.model:
            raise ValueError("Model already initialized")
        if not self.fabric:
            raise RuntimeError("Fabric instance must be provided to Inference for multi-GPU support.")

        self.model = StateEmbeddingModel.load_from_checkpoint(checkpoint, dropout=0.0, strict=False)
        self.model = self.fabric.setup_module(self.model)

        # get_embeddings will now receive the OmegaConf object.
        # It's assumed get_embeddings and get_embedding_cfg can correctly parse this.
        all_pe = self.protein_embeds or get_embeddings(self._vci_conf)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding = self.fabric.setup_module(self.model.pe_embedding)

        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        if self.protein_embeds is None:
            # get_embedding_cfg should be able to parse self._vci_conf (OmegaConf object)
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf).all_embeddings, weights_only=False)

    def init_from_model(self, model, protein_embeds=None):
        """
        Intended for use during training
        """
        if not self.fabric:
            raise RuntimeError("Fabric instance must be provided to Inference for multi-GPU support.")

        self.model = self.fabric.setup_module(model)
        if protein_embeds:
            self.protein_embeds = protein_embeds
        else:
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf).all_embeddings, weights_only=False)
        if hasattr(self.model, 'pe_embedding') and isinstance(self.model.pe_embedding, nn.Embedding):
            self.model.pe_embedding = self.fabric.setup_module(nn.Embedding.from_pretrained(self.protein_embeds))

    def get_gene_embedding(self, genes):
        protein_embeds = [self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(5120) for x in genes]
        protein_embeds = torch.stack(protein_embeds)
        protein_embeds = self.fabric.to_device(protein_embeds)
        return self.model.gene_embedding_layer(protein_embeds)

    def encode(self, dataloader, rda=None):
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                torch.cuda.empty_cache()
                _, _, _, emb, ds_emb = self.model._compute_embedding_for_batch(batch)
                embeddings = emb.detach().cpu().float().numpy()

                ds_emb = self.model.dataset_embedder(ds_emb)
                ds_embeddings = ds_emb.detach().cpu().float().numpy()

                yield embeddings, ds_embeddings

    def encode_adata(
        self,
        input_adata_path: str,
        output_adata_path: str | None = None,
        emb_key: str = "X_emb",
        dataset_name: str | None = None,
        batch_size: int = 32,
        lancedb_path: str | None = None,
        update_lancedb: bool = False,
        lancedb_batch_size: int = 1000,
    ):
        if not self.fabric:
            raise RuntimeError("Fabric instance must be provided to Inference for multi-GPU support.")

        shape_dict = self.__load_dataset_meta(input_adata_path)
        adata = anndata.read_h5ad(input_adata_path)
        if dataset_name is None:
            dataset_name = Path(input_adata_path).stem

        adata = self._convert_to_csr(adata)

        precision_dtype = get_precision_config(device_type=self.fabric.device.type)

        dataloader = create_dataloader(
            self._vci_conf, # This is the OmegaConf object
            adata=adata,
            adata_name=dataset_name or "inference",
            shape_dict=shape_dict,
            data_dir=os.path.dirname(input_adata_path),
            shuffle=False,
            protein_embeds=self.protein_embeds,
            precision=precision_dtype,
        )

        dataloader = self.fabric.setup_dataloader(dataloader)

        all_embeddings = []
        all_ds_embeddings = []

        iterable = tqdm(self.encode(dataloader), total=len(dataloader), desc="Encoding")
        if self.fabric.global_rank != 0:
             iterable = self.encode(dataloader)

        for embeddings, ds_embeddings in iterable:
            all_embeddings.append(embeddings)
            if ds_embeddings is not None:
                all_ds_embeddings.append(ds_embeddings)
        
        if self.fabric.world_size > 1:
            all_embeddings_tensor = torch.from_numpy(np.concatenate(all_embeddings, axis=0)).to(self.fabric.device)
            gathered_embeddings = self.fabric.all_gather(all_embeddings_tensor)
            all_embeddings = torch.cat(gathered_embeddings, dim=0).cpu().numpy()

            if len(all_ds_embeddings) > 0:
                all_ds_embeddings_tensor = torch.from_numpy(np.concatenate(all_ds_embeddings, axis=0)).to(self.fabric.device)
                gathered_ds_embeddings = self.fabric.all_gather(all_ds_embeddings_tensor)
                all_ds_embeddings = torch.cat(gathered_ds_embeddings, dim=0).cpu().numpy()
        else:
            all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
            if len(all_ds_embeddings) > 0:
                all_ds_embeddings = np.concatenate(all_ds_embeddings, axis=0).astype(np.float32)

        if len(all_ds_embeddings) > 0:
            all_embeddings = np.concatenate([all_embeddings, all_ds_embeddings], axis=-1)
        
        if output_adata_path is not None and self.fabric.global_rank == 0:
            adata.obsm[emb_key] = all_embeddings
            adata.write_h5ad(output_adata_path)
            log.info(f"Embeddings saved to {output_adata_path}")
        elif output_adata_path is not None and self.fabric.global_rank != 0:
            log.info(f"Rank {self.fabric.global_rank}: Not saving adata, only rank 0 handles saving.")

        if lancedb_path is not None and self.fabric.global_rank == 0:
            from .vectordb import StateVectorDB
        
            log.info(f"Saving embeddings to LanceDB at {lancedb_path}")
            vector_db = StateVectorDB(lancedb_path)
        
            metadata = adata.obs.copy()
        
            vector_db.create_or_update_table(
                embeddings=all_embeddings,
                metadata=metadata,
                embedding_key=emb_key,
                dataset_name=dataset_name or Path(input_adata_path).stem,
                batch_size=lancedb_batch_size
            )
        
            log.info(f"Successfully saved {len(all_embeddings)} embeddings to LanceDB")
        elif lancedb_path is not None and self.fabric.global_rank != 0:
            log.info(f"Rank {self.fabric.global_rank}: Not saving to LanceDB, only rank 0 handles saving.")

    def _convert_to_csr(self, adata):
        from scipy.sparse import csr_matrix, issparse
        if issparse(adata.X) and not isinstance(adata.X, csr_matrix):
            log.info(f"Converting {type(adata.X).__name__} to csr_matrix format")
            adata.X = csr_matrix(adata.X)
        return adata

    def decode_from_file(self, adata_path, emb_key: str, read_depth=None, batch_size=64):
        adata = anndata.read_h5ad(adata_path)
        genes = adata.var.index
        yield from self.decode_from_adata(adata, genes, emb_key, read_depth, batch_size)

    @torch.no_grad()
    def decode_from_adata(self, adata, genes, emb_key: str, read_depth=None, batch_size=64):
        if not self.fabric:
            raise RuntimeError("Fabric instance must be provided to Inference for multi-GPU support.")

        try:
            cell_embs = adata.obsm[emb_key]
        except:
            cell_embs = adata.X
        
        cell_embs = torch.Tensor(cell_embs)
        cell_embs = self.fabric.to_device(cell_embs)

        # Access use_rda from the OmegaConf config (self._vci_conf)
        use_rda = self._vci_conf.model.get("use_rda", False) # Use .get for robustness
        if use_rda and read_depth is None:
            read_depth = self._vci_conf.model.get("read_depth", 1000.0) # Use read_depth from config

        gene_embeds = self.get_gene_embedding(genes)
        
        for i in tqdm(range(0, cell_embs.size(0), batch_size), total=int(cell_embs.size(0) // batch_size)):
            cell_embeds_batch = cell_embs[i : i + batch_size]
            if use_rda:
                task_counts = torch.full((cell_embeds_batch.shape[0],), read_depth, device=self.fabric.device, dtype=self.fabric.precision)
            else:
                task_counts = None
            merged_embs = StateEmbeddingModel.resize_batch(cell_embeds_batch, gene_embeds, task_counts)
            logprobs_batch = self.model.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().float().numpy()
            yield logprobs_batch.squeeze()


def run_inference_cli(cfg: DictConfig, cli_args: argparse.Namespace):
    # 1. Initialize Fabric
    fabric = Fabric(
        accelerator=cli_args.accelerator,
        devices=cli_args.devices,
        strategy=cli_args.strategy,
        precision=cli_args.precision
    )
    fabric.launch()

    # Configure logging for each process
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - Rank {fabric.global_rank} - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    log.info(f"Fabric initialized on device: {fabric.device}, Global Rank: {fabric.global_rank}, World Size: {fabric.world_size}")

    # 3. Instantiate Inference class with Fabric and the OmegaConf object
    inference_pipeline = Inference(cfg=cfg, fabric=fabric)

    # 4. Load the model checkpoint
    checkpoint_path = cfg.model.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    fabric.barrier() # Ensure all processes see the file exists before loading
    inference_pipeline.load_model(checkpoint_path)
    log.info(f"Model loaded from {checkpoint_path} on rank {fabric.global_rank}.")

    # 5. Load protein embeddings
    protein_embeddings_path = cfg.embedding.protein_embeddings_path
    if not os.path.exists(protein_embeddings_path):
        raise FileNotFoundError(f"Protein embeddings file not found: {protein_embeddings_path}")
    # protein_embeds is loaded by `load_model` implicitly via `get_embeddings` and `get_embedding_cfg`
    # We can also load it explicitly and pass if `get_embeddings` is slow/complex
    # For now, rely on `load_model`'s internal logic as before.
    log.info(f"Protein embeddings configured from {protein_embeddings_path} on rank {fabric.global_rank}.")

    # 6. Run encoding (inference)
    input_adata_path = cli_args.input_adata_path
    output_adata_path = cli_args.output_adata_path
    emb_key = cfg.inference.emb_key
    dataset_name = cfg.inference.dataset_name
    batch_size = cfg.inference.batch_size
    lancedb_path = cfg.inference.lancedb_path
    update_lancedb = cfg.inference.update_lancedb
    lancedb_batch_size = cfg.inference.lancedb_batch_size


    if not os.path.exists(input_adata_path):
        raise FileNotFoundError(f"Input AnnData file not found: {input_adata_path}")

    log.info(f"Starting encoding on rank {fabric.global_rank}...")
    inference_pipeline.encode_adata(
        input_adata_path=input_adata_path,
        output_adata_path=output_adata_path,
        emb_key=emb_key,
        dataset_name=dataset_name,
        batch_size=batch_size,
        lancedb_path=lancedb_path,
        update_lancedb=update_lancedb,
        lancedb_batch_size=lancedb_batch_size,
    )
    log.info(f"Encoding finished on rank {fabric.global_rank}.")

    # 7. (Optional) Run decoding (inference)
    if cfg.inference.run_decode and fabric.global_rank == 0:
        log.info("Starting decoding on rank 0...")
        decoded_adata_path = output_adata_path if output_adata_path and os.path.exists(output_adata_path) else input_adata_path
        read_depth = cfg.model.get("read_depth", 1000.0) # Default if not in config
        
        for batch_probs in inference_pipeline.decode_from_file(
            decoded_adata_path, emb_key=emb_key, read_depth=read_depth, batch_size=batch_size
        ):
            # Process decoded probabilities
            pass
        log.info("Decoding finished on rank 0.")

    # Clean up files - this cleanup should be manual or part of a separate script
    # as we are now working with actual files, not dummies.
    # The `cleanup_dummies` argument is removed since we don't create dummies.
    # If you want to delete output files, you'd need a specific `--delete_output` arg.


def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU inference for StateEmbeddingModel using PyTorch Lightning Fabric.")

    # Argument for the OmegaConf configuration file
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the OmegaConf YAML configuration file.")
    parser.add_argument("--input_adata_path",type=Path,required=True,
                        help="Path to Input .h5 file")
    
    parser.add_argument("--output_adata_path",type=Path,required=True,
                        help="Path to where output .h5 file is to be written") 
    # Fabric Parameters (can be overridden from CLI)
    parser.add_argument("--accelerator", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"],
                        help="Type of accelerator to use (e.g., 'cuda', 'cpu', 'mps').")
    parser.add_argument("--devices", type=str, default="auto",
                        help="Number of devices to use (e.g., 'auto', 1, 2, '0,1').")
    parser.add_argument("--strategy", type=str, default="ddp",
                        choices=["auto", "ddp", "ddp_spawn", "fsdp", "deepspeed"],
                        help="Distributed training strategy (e.g., 'ddp', 'fsdp').")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32-true", "16-mixed", "bf16-mixed"],
                        help="Precision for training (e.g., 'bf16-mixed', '32-true').")
    
    # Optional overrides for common inference parameters
    # These will override values specified in the config file
    parser.add_argument("--batch_size", type=int,
                        help="Override batch size from config file.")
    parser.add_argument("--emb_key", type=str,
                        help="Override embedding key from config file.")
    parser.add_argument("--dataset_name", type=str,
                        help="Override dataset name from config file.")
    parser.add_argument("--lancedb_path", type=str,
                        help="Override LanceDB path from config file (set to null in config to disable by default).")
    parser.add_argument("--run_decode", action="store_true",
                        help="Force running the decoding step (overrides config setting).")


    cli_args = parser.parse_args()

    # Load the base configuration from the specified YAML file
    cfg = OmegaConf.load(cli_args.config)
    for key in cfg.keys():
        print(f"Key : {key} , Value : {cfg[key]} ")

    # Apply CLI overrides to the OmegaConf object
    # This creates a merged view where CLI args take precedence
    if cli_args.batch_size is not None:
        cfg.inference.batch_size = cli_args.batch_size
    if cli_args.emb_key is not None:
        cfg.inference.emb_key = cli_args.emb_key
    if cli_args.dataset_name is not None:
        cfg.inference.dataset_name = cli_args.dataset_name
    if cli_args.lancedb_path is not None:
        cfg.inference.lancedb_path = cli_args.lancedb_path
    if cli_args.run_decode:
        cfg.inference.run_decode = True # Set to True if flag is present

    # Pass the OmegaConf object and original CLI args to the runner function
    run_inference_cli(cfg, cli_args)

if __name__ == "__main__":
    main()