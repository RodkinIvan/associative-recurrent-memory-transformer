"""
Custom TrainerCallback to handle DeepSpeed ZeRO-3 checkpoint consolidation and Hub pushing.

When using DeepSpeed Stage 3, model weights are sharded across GPUs and stored in
global_step{N}/ directories. The Hugging Face Trainer's default _push_from_checkpoint
doesn't handle this format, so we need to consolidate the checkpoint before pushing.

Additionally, this callback inlines custom ARMT modeling code into a single file
for easy loading from the Hub with trust_remote_code=True.
"""

import os
import shutil
import json
from pathlib import Path
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch
import logging

logger = logging.getLogger(__name__)


class DeepSpeedCheckpointCallback(TrainerCallback):
    """
    Callback to consolidate DeepSpeed ZeRO-3 checkpoints before pushing to Hub.
    
    This callback:
    1. Detects when a checkpoint is saved with DeepSpeed ZeRO-3
    2. Consolidates the sharded weights into a single pytorch_model.bin
    3. Inlines custom ARMT modeling code into modeling_armt.py
    4. Updates config.json with auto_map for custom code
    5. Allows the Trainer to push the consolidated checkpoint to Hub
    """
    
    def __init__(self, consolidate_on_save=True, modeling_code_dir=None, model_class_name=None):
        """
        Args:
            consolidate_on_save: If True, consolidate checkpoint immediately after saving.
                                If False, only consolidate before pushing to Hub.
            modeling_code_dir: Path to directory containing custom modeling code (e.g., "modeling_amt").
                             If None, will try to auto-detect.
            model_class_name: Name of the model class (e.g., "InnerLoopARMTForCausalLM").
                            If None, will try to infer from config.
        """
        self.consolidate_on_save = consolidate_on_save
        self.modeling_code_dir = Path(modeling_code_dir) if modeling_code_dir else None
        self.model_class_name = model_class_name
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called after a checkpoint is saved. Consolidates DeepSpeed checkpoint if needed.
        """
        # Only run on main process
        if not args.should_save:
            return control
        
        # Check if we're using DeepSpeed
        if args.deepspeed is None:
            return control
        
        # Get the checkpoint directory that was just saved
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        
        if not os.path.exists(checkpoint_folder):
            logger.warning(f"Checkpoint folder {checkpoint_folder} not found, skipping consolidation")
            return control
        
        # Check if this is a DeepSpeed ZeRO-3 checkpoint (has global_step* directory)
        global_step_dirs = [d for d in os.listdir(checkpoint_folder) if d.startswith('global_step')]
        
        if not global_step_dirs:
            logger.debug(f"No DeepSpeed checkpoint found in {checkpoint_folder}, skipping consolidation")
            return control
        
        # Check if already consolidated
        consolidated_file = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(consolidated_file):
            logger.debug(f"Checkpoint already consolidated: {consolidated_file}")
            return control
        
        # Consolidate the checkpoint
        if self.consolidate_on_save or args.push_to_hub:
            logger.info(f"Consolidating DeepSpeed ZeRO-3 checkpoint at {checkpoint_folder}")
            try:
                self._consolidate_checkpoint(checkpoint_folder, state.global_step)
                logger.info(f"✓ Successfully consolidated checkpoint to {consolidated_file}")
            except Exception as e:
                logger.error(f"Failed to consolidate DeepSpeed checkpoint: {e}")
                import traceback
                traceback.print_exc()
        
        # Inline custom modeling code if pushing to Hub
        if args.push_to_hub:
            try:
                self._inline_modeling_code(checkpoint_folder)
                logger.info(f"✓ Successfully inlined custom modeling code")
            except Exception as e:
                logger.warning(f"Failed to inline modeling code (checkpoint will still work, but may need local code): {e}")
                import traceback
                traceback.print_exc()
        
        return control
    
    def _consolidate_checkpoint(self, checkpoint_dir, global_step):
        """
        Consolidate a DeepSpeed ZeRO-3 checkpoint into a single pytorch_model.bin file.
        
        This uses the DeepSpeed checkpoint utilities to merge the sharded weights.
        """
        checkpoint_path = Path(checkpoint_dir)
        global_step_dir = checkpoint_path / f"global_step{global_step}"
        
        if not global_step_dir.exists():
            raise ValueError(f"global_step directory not found: {global_step_dir}")
        
        # Import DeepSpeed utilities
        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        except ImportError:
            logger.warning("DeepSpeed not found, trying alternative consolidation method")
            self._consolidate_checkpoint_manual(checkpoint_dir, global_step)
            return
        
        # Use DeepSpeed's utility to get the consolidated state dict
        logger.info(f"Loading DeepSpeed checkpoint from {checkpoint_dir}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_dir))
        
        # Save the consolidated state dict
        output_file = checkpoint_path / "pytorch_model.bin"
        logger.info(f"Saving consolidated checkpoint to {output_file}")
        torch.save(state_dict, output_file)
        
        logger.info(f"Consolidated checkpoint size: {output_file.stat().st_size / (1024**3):.2f} GB")
    
    def _consolidate_checkpoint_manual(self, checkpoint_dir, global_step):
        """
        Fallback: manually consolidate checkpoint by running zero_to_fp32.py script.
        """
        checkpoint_path = Path(checkpoint_dir)
        zero_to_fp32_script = checkpoint_path / "zero_to_fp32.py"
        
        if not zero_to_fp32_script.exists():
            raise ValueError(f"zero_to_fp32.py not found in {checkpoint_dir}")
        
        output_file = checkpoint_path / "pytorch_model.bin"
        
        # Run the consolidation script
        import subprocess
        logger.info(f"Running zero_to_fp32.py to consolidate checkpoint")
        
        cmd = [
            "python",
            str(zero_to_fp32_script),
            str(checkpoint_dir),
            str(output_file),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"zero_to_fp32.py failed: {result.stderr}")
        
        if not output_file.exists():
            raise RuntimeError(f"Consolidation failed: {output_file} was not created")
        
        logger.info(f"Successfully consolidated checkpoint to {output_file}")
    
    def _inline_modeling_code(self, checkpoint_dir):
        """
        Inline custom ARMT modeling code into a single modeling_armt.py file.
        This allows the model to be loaded from Hub with trust_remote_code=True
        without requiring the local modeling_amt package.
        """
        checkpoint_path = Path(checkpoint_dir)
        
        # Auto-detect modeling code directory if not provided
        if self.modeling_code_dir is None:
            # Try to find modeling_amt in common locations
            possible_dirs = [
                Path.cwd() / "modeling_amt",
                Path(__file__).parent / "modeling_amt",
                Path(__file__).parent.parent / "modeling_amt",
            ]
            for d in possible_dirs:
                if d.exists() and (d / "model.py").exists():
                    self.modeling_code_dir = d
                    break
            
            if self.modeling_code_dir is None:
                logger.warning("Could not auto-detect modeling_amt directory, skipping code inlining")
                return
        
        modeling_dir = Path(self.modeling_code_dir)
        if not modeling_dir.exists():
            logger.warning(f"Modeling directory not found: {modeling_dir}, skipping code inlining")
            return
        
        logger.info(f"Inlining modeling code from {modeling_dir}")
        
        # Read all ARMT modeling files
        files_to_inline = {
            "act_utils.py": modeling_dir / "act_utils.py",
            "language_modeling.py": modeling_dir / "language_modeling.py",
            "model.py": modeling_dir / "model.py",
            "inner_loop.py": modeling_dir / "inner_loop.py",
        }
        
        # Read and process each file
        code_sections = []
        for name, path in files_to_inline.items():
            if not path.exists():
                logger.debug(f"Skipping {name} (not found)")
                continue
            
            code = path.read_text()
            
            # Remove internal imports to make code self-contained
            code = code.replace("from modeling_amt.act_utils import", "# inlined act_utils: removed import")
            code = code.replace("from modeling_amt.language_modeling import", "# inlined language_modeling: removed import")
            code = code.replace("from modeling_amt.model import ARMTConfig", "# inlined ARMTConfig: removed import")
            
            code_sections.append(f"# ---- {name} ----\n{code}\n")
        
        # Create single inlined file
        single_file = checkpoint_path / "modeling_armt.py"
        single_file.write_text(
            "# === Inlined ARMT for HF Hub (single-file) ===\n"
            "# This file contains all ARMT modeling code inlined for easy loading.\n"
            "# Generated automatically during training checkpoint save.\n\n"
            + "\n".join(code_sections)
        )
        
        logger.info(f"Created {single_file.name} with {len(code_sections)} code sections")
        
        # Update config.json to use the inlined code
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            
            # Determine model class name
            model_class = self.model_class_name
            if model_class is None:
                # Try to infer from architectures in config
                if "architectures" in config and config["architectures"]:
                    model_class = config["architectures"][0]
                else:
                    # Default to InnerLoopARMTForCausalLM
                    model_class = "InnerLoopARMTForCausalLM"
                    logger.info(f"Using default model class: {model_class}")
            
            # Update config with auto_map
            config["architectures"] = [model_class]
            config["auto_map"] = {
                "AutoConfig": "modeling_armt.ARMTConfig",
                "AutoModelForCausalLM": f"modeling_armt.{model_class}",
            }
            
            # Save updated config
            config_path.write_text(json.dumps(config, indent=2))
            logger.info(f"Updated config.json with auto_map for {model_class}")
        else:
            logger.warning("config.json not found, could not update auto_map")


class SafeDeepSpeedCheckpointCallback(TrainerCallback):
    """
    A safer version that only consolidates on-demand before Hub push.
    
    This version doesn't consolidate every checkpoint (saves disk space),
    but ensures consolidation happens before pushing to Hub.
    """
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Only consolidate if we're about to push to Hub."""
        # Only run on main process
        if not args.should_save:
            return control
        
        # Only consolidate if we're pushing to Hub
        if not args.push_to_hub:
            return control
        
        # Check if it's time to push (based on hub_strategy)
        from transformers.trainer_utils import HubStrategy
        
        should_push = False
        if args.hub_strategy == HubStrategy.EVERY_SAVE:
            should_push = True
        elif args.hub_strategy == HubStrategy.CHECKPOINT:
            # Push on checkpoint (handled by Trainer)
            should_push = True
        elif args.hub_strategy == HubStrategy.END:
            # Don't push during training
            should_push = False
        
        if not should_push:
            return control
        
        # Use the consolidation logic from DeepSpeedCheckpointCallback
        callback = DeepSpeedCheckpointCallback(consolidate_on_save=False)
        return callback.on_save(args, state, control, **kwargs)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Consolidate final checkpoint if pushing to Hub at end of training."""
        if args.push_to_hub and args.hub_strategy == "end":
            # Consolidate the final checkpoint
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            callback = DeepSpeedCheckpointCallback(consolidate_on_save=True)
            return callback.on_save(args, state, control, **kwargs)
        return control

