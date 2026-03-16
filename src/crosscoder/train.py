import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .dataset import PairedActivationDataset, collate_activations, create_paired_activation_dataset
from .model import SPARCCrossCoder, create_crosscoder
from .utils import (
    get_checkpoint_dir,
    get_device,
    get_metrics_dir,
    get_results_dir,
    save_checkpoint,
    save_json,
    set_seed,
)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


class QualityGateError(Exception):
    pass


def check_quality_gate(
    fve_u: float,
    fve_c: float,
    dead_neuron_fraction: float,
    epoch: int,
) -> None:
    if fve_u < config.FVE_THRESHOLD:
        raise QualityGateError(
            f"Quality gate failed at epoch {epoch}: FVE_u={fve_u:.4f} < {config.FVE_THRESHOLD}"
        )
    if fve_c < config.FVE_THRESHOLD:
        raise QualityGateError(
            f"Quality gate failed at epoch {epoch}: FVE_c={fve_c:.4f} < {config.FVE_THRESHOLD}"
        )
    if dead_neuron_fraction > config.DEAD_NEURON_THRESHOLD:
        raise QualityGateError(
            f"Quality gate failed at epoch {epoch}: dead_neurons={dead_neuron_fraction:.4f} > {config.DEAD_NEURON_THRESHOLD}"
        )


def train_crosscoder(
    activations_data: Dict,
    model_name: str,
    method: str,
    component: str,
    token_type: str,
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    checkpoint_every: int = config.CHECKPOINT_EVERY,
    results_dir: Optional[Path] = None,
    expansion_factor: Optional[int] = None,
    topk: Optional[int] = None,
) -> Dict:
    set_seed()
    device = get_device()
    
    if results_dir is None:
        results_dir = get_results_dir(model_name, method, component, token_type)
    checkpoint_dir = get_checkpoint_dir(results_dir)
    metrics_dir = get_metrics_dir(results_dir)
    
    train_dataset = create_paired_activation_dataset(activations_data, split="train")
    val_dataset = create_paired_activation_dataset(activations_data, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_activations,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_activations,
    )
    
    crosscoder = create_crosscoder(
        model_name, component, token_type,
        expansion_factor=expansion_factor,
        topk=topk,
    )
    crosscoder = crosscoder.to(device)
    
    optimizer = AdamW(crosscoder.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * config.WARMUP_FRACTION)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    training_history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_fve_u": [],
        "train_fve_c": [],
        "val_fve_u": [],
        "val_fve_c": [],
        "dead_neurons": [],
        "l0_u": [],
        "l0_c": [],
        "self_recon": [],
        "cross_recon": [],
        "sparsity": [],
    }
    
    print(f"\nTraining cross-coder for {model_name}/{method}/{component}/{token_type}")
    print(f"  Dict size: {crosscoder.dict_size}, TopK: {crosscoder.topk}")
    print(f"  Forced shared: {crosscoder.n_forced_shared} ({crosscoder.forced_shared_fraction*100:.1f}%)")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        crosscoder.train()
        train_losses = []
        train_fve_u_list = []
        train_fve_c_list = []
        train_dead_list = []
        train_l0_u_list = []
        train_l0_c_list = []
        train_self_recon = []
        train_cross_recon = []
        train_sparsity = []
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, unit="batch")
        
        for batch in batch_pbar:
            x_u = batch["activations_u"].to(device, dtype=torch.float32)
            x_c = batch["activations_c"].to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            outputs = crosscoder(x_u, x_c)
            losses = crosscoder.compute_loss(x_u, x_c, outputs)
            
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(crosscoder.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            crosscoder.sync_forced_shared_post_step()
            
            with torch.no_grad():
                fve = crosscoder.compute_fve(x_u, x_c, outputs)
                dead = crosscoder.compute_dead_neurons(outputs["z_u"], outputs["z_c"])
                l0 = crosscoder.compute_l0_sparsity(outputs["z_u"], outputs["z_c"])
            
            train_losses.append(losses["total"].item())
            train_fve_u_list.append(fve["fve_u"])
            train_fve_c_list.append(fve["fve_c"])
            train_dead_list.append(dead)
            train_l0_u_list.append(l0["l0_u"])
            train_l0_c_list.append(l0["l0_c"])
            train_self_recon.append(losses["self_recon"].item())
            train_cross_recon.append(losses["cross_recon"].item())
            train_sparsity.append(losses["sparsity"].item())
            
            batch_pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "fve_u": f"{fve['fve_u']:.3f}",
                "fve_c": f"{fve['fve_c']:.3f}",
            })
        
        crosscoder.eval()
        val_losses = []
        val_fve_u_list = []
        val_fve_c_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                x_u = batch["activations_u"].to(device, dtype=torch.float32)
                x_c = batch["activations_c"].to(device, dtype=torch.float32)
                
                outputs = crosscoder(x_u, x_c)
                losses = crosscoder.compute_loss(x_u, x_c, outputs)
                fve = crosscoder.compute_fve(x_u, x_c, outputs)
                
                val_losses.append(losses["total"].item())
                val_fve_u_list.append(fve["fve_u"])
                val_fve_c_list.append(fve["fve_c"])
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_train_fve_u = sum(train_fve_u_list) / len(train_fve_u_list)
        avg_train_fve_c = sum(train_fve_c_list) / len(train_fve_c_list)
        avg_val_fve_u = sum(val_fve_u_list) / len(val_fve_u_list) if val_fve_u_list else 0
        avg_val_fve_c = sum(val_fve_c_list) / len(val_fve_c_list) if val_fve_c_list else 0
        avg_dead = sum(train_dead_list) / len(train_dead_list)
        avg_l0_u = sum(train_l0_u_list) / len(train_l0_u_list)
        avg_l0_c = sum(train_l0_c_list) / len(train_l0_c_list)
        avg_self_recon = sum(train_self_recon) / len(train_self_recon)
        avg_cross_recon = sum(train_cross_recon) / len(train_cross_recon)
        avg_sparsity = sum(train_sparsity) / len(train_sparsity)
        
        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["train_fve_u"].append(avg_train_fve_u)
        training_history["train_fve_c"].append(avg_train_fve_c)
        training_history["val_fve_u"].append(avg_val_fve_u)
        training_history["val_fve_c"].append(avg_val_fve_c)
        training_history["dead_neurons"].append(avg_dead)
        training_history["l0_u"].append(avg_l0_u)
        training_history["l0_c"].append(avg_l0_c)
        training_history["self_recon"].append(avg_self_recon)
        training_history["cross_recon"].append(avg_cross_recon)
        training_history["sparsity"].append(avg_sparsity)
        
        epoch_pbar.set_postfix({
            "train": f"{avg_train_loss:.4f}",
            "val": f"{avg_val_loss:.4f}",
            "fve_u": f"{avg_val_fve_u:.3f}",
            "fve_c": f"{avg_val_fve_c:.3f}",
            "dead": f"{avg_dead:.3f}",
        })
        
        if (epoch + 1) % checkpoint_every == 0:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "fve_u": avg_val_fve_u,
                "fve_c": avg_val_fve_c,
                "dead_neurons": avg_dead,
            }
            save_checkpoint(crosscoder, optimizer, epoch + 1, metrics, checkpoint_dir)
    
    final_metrics = {
        "epoch": num_epochs,
        "train_loss": training_history["train_loss"][-1],
        "val_loss": training_history["val_loss"][-1],
        "fve_u": training_history["val_fve_u"][-1],
        "fve_c": training_history["val_fve_c"][-1],
        "dead_neurons": training_history["dead_neurons"][-1],
        "l0_u": training_history["l0_u"][-1],
        "l0_c": training_history["l0_c"][-1],
    }
    save_checkpoint(crosscoder, optimizer, num_epochs, final_metrics, checkpoint_dir, is_final=True)
    
    save_json(training_history, metrics_dir / "training_metrics.json")
    
    final_fve_u = training_history["val_fve_u"][-1]
    final_fve_c = training_history["val_fve_c"][-1]
    final_dead = training_history["dead_neurons"][-1]
    
    check_quality_gate(final_fve_u, final_fve_c, final_dead, num_epochs)
    
    print(f"\nTraining complete!")
    print(f"  Final FVE_u: {final_fve_u:.4f}, FVE_c: {final_fve_c:.4f}")
    print(f"  Dead neurons: {final_dead:.4f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    
    return {
        "crosscoder": crosscoder,
        "training_history": training_history,
        "final_metrics": final_metrics,
        "results_dir": results_dir,
    }


def load_trained_crosscoder(
    model_name: str,
    method: str,
    component: str,
    token_type: str,
) -> SPARCCrossCoder:
    device = get_device()
    results_dir = get_results_dir(model_name, method, component, token_type)
    checkpoint_dir = get_checkpoint_dir(results_dir)
    
    crosscoder = create_crosscoder(model_name, component, token_type)
    crosscoder = crosscoder.to(device)
    
    checkpoint_path = checkpoint_dir / "final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    crosscoder.load_state_dict(checkpoint["model_state_dict"])
    crosscoder.eval()
    
    return crosscoder


def compute_all_feature_activations(
    crosscoder: SPARCCrossCoder,
    activations_data: Dict,
) -> Dict:
    device = get_device()
    crosscoder.eval()
    
    activations_u = activations_data["activations_u"].to(device, dtype=torch.float32)
    activations_c = activations_data["activations_c"].to(device, dtype=torch.float32)
    
    all_z_u = []
    all_z_c = []
    
    batch_size = 256
    num_samples = activations_u.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            x_u = activations_u[i:i+batch_size]
            x_c = activations_c[i:i+batch_size]
            
            outputs = crosscoder(x_u, x_c)
            all_z_u.append(outputs["z_u"].cpu())
            all_z_c.append(outputs["z_c"].cpu())
    
    return {
        "z_u": torch.cat(all_z_u, dim=0),
        "z_c": torch.cat(all_z_c, dim=0),
        "sample_ids": activations_data["sample_ids"],
        "image_types": activations_data["image_types"],
        "splits": activations_data["splits"],
    }
