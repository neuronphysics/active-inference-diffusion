"""
Logging utilities
"""

import wandb
from typing import Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
import torch
class Logger:
    """
    Unified logger supporting console, file, and wandb
    """
    
    def __init__(
        self,
        use_wandb: bool = True,
        project_name: str = "active-inference",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs"
    ):
        self.use_wandb = use_wandb
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config
            )
            
        # Create local log file
        self.log_file = self.log_dir / f"{experiment_name or 'experiment'}.jsonl"
        
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics"""
        processed_metrics = {}
        for key, value in metrics.items():
            # Handle PyTorch tensors
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # Scalar tensor
                    processed_metrics[key] = value.item()
                else:
                    processed_metrics[key] = value.cpu().detach().tolist()
            # Handle numpy arrays (if any)
            elif isinstance(value, np.ndarray):
                processed_metrics[key] = value.tolist()
            else:
                processed_metrics[key] = value
    
        processed_metrics['step'] = step
    
        # Log to wandb
        if self.use_wandb:
            wandb.log(processed_metrics, step=step)
    
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(processed_metrics) + '\n')
            
    def log_video(self, video: np.ndarray, caption: str, step: int):
        """Log video to wandb"""
        if self.use_wandb:
            wandb.log({
                caption: wandb.Video(video, fps=30, format="mp4")
            }, step=step)
            
    def finish(self):
        """Finish logging"""
        if self.use_wandb:
            wandb.finish()


