import torch
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from config import Config
from data_loader import MAGDataLoader
from models.foundational import FoundationalModel
from models.mag_model import MAGModel
from train import Trainer
from eval import Evaluator
from encoders.wfvae import WFVAE
from encoders.text_encoder import TextEncoder

def load_config(stage: int) -> Dict[str, Any]:
    """Load YAML configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    return full_config['training'][f'stage{stage}']

def setup_model(stage: int, config: Config, device: torch.device) -> torch.nn.Module:
    """Initialize model based on training stage"""
    encoder_levels = config.config_data['encoders']['video'].get('levels', 3)
    wfvae = WFVAE(levels=encoder_levels)
    
    if stage == 1:
        mt5 = TextEncoder(model_type='mT5')
        clip = TextEncoder(model_type='CLIP')
        return FoundationalModel(wfvae, mt5, clip, config).to(device)
    
    # For stages 2 and 3 use MAGModel
    model = MAGModel(config.config_data).to(device)
    
    if stage == 3:
        # Freeze backbone weights for Stage 3
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen for Stage 3 training")
    
    return model

def train_stage(stage: int, device: torch.device) -> None:
    """Execute full training pipeline for one stage"""
    print(f"\n{'='*40}")
    print(f"STARTING STAGE {stage} TRAINING")
    print(f"{'='*40}\n")
    
    # Initialize configuration
    stage_config = load_config(stage)
    config = Config(stage)
    hyperparams = config.get_hyperparams()
    
    print(f"Configuration loaded: {hyperparams}")
    print(f"Using device: {device}")
    
    # Setup model and data loader
    model = setup_model(stage, config, device)
    data_loader = MAGDataLoader(stage, config)
    
    # Initialize and run trainer
    trainer = Trainer(stage, config, device)
    trainer.model = model  # Attach created model to trainer
    loss_history = trainer.train(hyperparams.get("steps", 100000))
    
    # Save final checkpoint
    checkpoint_dir = config.config_data['training'].get('checkpoint_dir', 'checkpoints')
    Path(checkpoint_dir).mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'stage': stage,
        'loss_history': loss_history
    }, f"{checkpoint_dir}/stage{stage}_final.pt")
    print(f"Stage {stage} training completed. Model saved")

def run_evaluation(config: Config, device: torch.device) -> None:
    """Execute full evaluation pipeline"""
    print("\n\nStarting Evaluation Phase")
    print("=" * 40)
    
    # Initialize model for stage 3 (final model)
    model = setup_model(3, Config(3), device)
    
    # Load trained weights
    checkpoint_dir = config.config_data['training'].get('checkpoint_dir', 'checkpoints')
    checkpoint = torch.load(f"{checkpoint_dir}/stage3_final.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run evaluation
    evaluator = Evaluator(model, config, device)
    results = evaluator.run_metrics()
    
    # Print and save results
    print("\nEvaluation Results:")
    print("-" * 60)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join([f"{k}: {v:.4f}" for k, v in results.items()]))
    
    # Generate qualitative visualizations
    eval_output_dir = config.config_data['evaluation'].get('output_dir', 'results')
    sample_video = evaluator._generate_sample_video()
    evaluator.render_fps_elements(sample_video, eval_output_dir)
    print(f"Qualitative results saved to {eval_output_dir}")

def main() -> None:
    """Main orchestration function for MAG system"""
    parser = argparse.ArgumentParser(description="MAG Training Pipeline")
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], 
                        help="Training stage to execute (1-3)")
    parser.add_argument('--eval', action='store_true', 
                        help="Run evaluation after training")
    parser.add_argument('--device', type=str, default="cuda",
                        help="Device for training (cuda or cpu)")
    args = parser.parse_args()

    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Execute requested training stage
    if args.stage:
        train_stage(args.stage, device)
    
    # Execute evaluation if requested
    if args.eval:
        run_evaluation(Config(3), device)

if __name__ == "__main__":
    main()
