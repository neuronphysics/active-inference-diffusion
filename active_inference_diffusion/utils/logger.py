"""
Logging utilities
"""

import wandb
from typing import Dict, Any, Optional
import json
from pathlib import Path


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
        # Add step to metrics
        metrics['step'] = step
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
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


