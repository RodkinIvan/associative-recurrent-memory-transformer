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
from huggingface_hub import HfApi, create_repo

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
        self.last_consolidated_checkpoint = None  # Track which checkpoint was consolidated
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called after a checkpoint is saved. Consolidates DeepSpeed checkpoint if needed.
        """
        # Only run on main process
        if not args.should_save:
            return control
        
        # Get the checkpoint directory that was just saved
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        
        logger.info(f"="*80)
        logger.info(f"DeepSpeedCheckpointCallback.on_save() called for checkpoint: {checkpoint_folder}")
        
        if not os.path.exists(checkpoint_folder):
            logger.warning(f"Checkpoint folder {checkpoint_folder} not found, skipping consolidation")
            return control
        
        # Log what's currently in the checkpoint directory
        try:
            files_in_checkpoint = os.listdir(checkpoint_folder)
            logger.info(f"Files in checkpoint before consolidation: {files_in_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not list checkpoint directory: {e}")
        
        # Check if this is a DeepSpeed ZeRO-3 checkpoint (has global_step* directory)
        global_step_dirs = [d for d in os.listdir(checkpoint_folder) if d.startswith('global_step')]
        
        if not global_step_dirs:
            logger.debug(f"No DeepSpeed checkpoint found in {checkpoint_folder}, skipping consolidation")
            return control
        
        # When global_step* dirs exist, the Trainer's saved model file is a ZeRO-3
        # stub that only contains non-partitioned params (e.g. layer norms, memory
        # tokens) while all sharded weights are size-0 placeholders.  Remove it so
        # we can replace it with the properly consolidated weights.
        # The stub may be pytorch_model.bin or model.safetensors depending on
        # save_safetensors setting.
        stub_files = [
            os.path.join(checkpoint_folder, "pytorch_model.bin"),
            os.path.join(checkpoint_folder, "pytorch_model.bin.index.json"),
            os.path.join(checkpoint_folder, "model.safetensors"),
            os.path.join(checkpoint_folder, "model.safetensors.index.json"),
        ]
        for stub_path in stub_files:
            if os.path.exists(stub_path):
                stub_size = os.path.getsize(stub_path)
                logger.info(f"Removing ZeRO-3 stub {os.path.basename(stub_path)} "
                            f"({stub_size / (1024**2):.2f} MB) before consolidation")
                os.remove(stub_path)
        
        # Consolidate the checkpoint
        if self.consolidate_on_save:
            logger.info(f"Consolidating DeepSpeed ZeRO-3 checkpoint at {checkpoint_folder}")
            try:
                self._consolidate_checkpoint(checkpoint_folder, state.global_step)
                logger.info(f"✓ Successfully consolidated checkpoint at {checkpoint_folder}")
            except Exception as e:
                logger.error(f"Failed to consolidate DeepSpeed checkpoint: {e}")
                import traceback
                traceback.print_exc()
        
        # Note: inlining code and pushing to hub are handled by PushToHubCallback
        
        # Log final state of checkpoint directory
        try:
            import glob
            
            files_after = os.listdir(checkpoint_folder)
            logger.info(f"Files in checkpoint after consolidation: {len(files_after)} files")
            
            checkpoint_path = Path(checkpoint_folder)
            safetensors_single = checkpoint_path / "model.safetensors"
            safetensors_index = checkpoint_path / "model.safetensors.index.json"
            safetensors_shards = glob.glob(str(checkpoint_path / "model-*.safetensors"))
            
            if safetensors_single.exists():
                size = safetensors_single.stat().st_size / (1024**3)
                logger.info(f"  ✓ model.safetensors: {size:.2f} GB")
            elif safetensors_index.exists() and safetensors_shards:
                total_size = sum(Path(f).stat().st_size for f in safetensors_shards) / (1024**3)
                logger.info(f"  ✓ {len(safetensors_shards)} safetensors shards, total: {total_size:.2f} GB")
            else:
                logger.warning(f"  ✗ No model files found after consolidation")
            
            # Check for config.json (modeling_armt.py is created later by PushToHubCallback)
            config_path = checkpoint_path / "config.json"
            if config_path.exists():
                size = config_path.stat().st_size / (1024**2)
                logger.info(f"  ✓ config.json: {size:.2f} MB")
            else:
                logger.warning(f"  ✗ config.json: NOT FOUND")
        except Exception as e:
            logger.warning(f"Could not verify checkpoint files: {e}")
        
        # Note: Hub push is handled by PushToHubCallback
        
        logger.info(f"="*80)
        
        # Track that we consolidated this checkpoint
        self.last_consolidated_checkpoint = checkpoint_folder
        
        return control
    
    def _consolidate_checkpoint(self, checkpoint_dir, global_step):
        """
        Consolidate a DeepSpeed ZeRO-3 checkpoint directly to bf16 safetensors.
        
        Uses DeepSpeed's Python API to gather sharded weights, casts to bf16,
        and saves via safetensors — no intermediate .bin files.
        """
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        from safetensors.torch import save_file
        
        checkpoint_path = Path(checkpoint_dir)
        global_step_dir = checkpoint_path / f"global_step{global_step}"
        
        if not global_step_dir.exists():
            raise ValueError(f"global_step directory not found: {global_step_dir}")
        
        logger.info(f"Gathering ZeRO-3 state dict from {checkpoint_dir} ...")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        logger.info(f"  Gathered {len(state_dict)} parameters")
        
        # Cast fp32 → bf16 (the training precision)
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor) and state_dict[key].is_floating_point():
                state_dict[key] = state_dict[key].to(torch.bfloat16)
        
        # Clone tensors that share memory (safetensors requires independent storage)
        ptr_to_names = {}
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                ptr = tensor.data_ptr()
                ptr_to_names.setdefault(ptr, []).append(name)
        for names in ptr_to_names.values():
            for dup in names[1:]:
                state_dict[dup] = state_dict[dup].clone()
        
        out_path = checkpoint_path / "model.safetensors"
        logger.info(f"Saving {len(state_dict)} parameters as bf16 safetensors → {out_path.name} ...")
        save_file(state_dict, str(out_path))
        
        size_gb = out_path.stat().st_size / (1024**3)
        logger.info(f"Saved model.safetensors ({size_gb:.2f} GB)")
        
        try:
            os.sync()
        except Exception:
            pass

class PushToHubCallback(TrainerCallback):
    """
    Callback to inline modeling code and push checkpoints to the Hub.
    Works with or without DeepSpeed.
    """
    def __init__(self, modeling_code_dir=None, model_class_name=None):
        self.modeling_code_dir = Path(modeling_code_dir) if modeling_code_dir else None
        self.model_class_name = model_class_name
        self.last_pushed_checkpoint = None
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Only run on main process
        if not args.should_save:
            return control
        
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        logger.info(f"PushToHubCallback.on_save() called for checkpoint: {checkpoint_folder}")
        if not os.path.exists(checkpoint_folder):
            logger.warning(f"Checkpoint folder {checkpoint_folder} not found, skipping push")
            return control
        
        # Inline modeling code
        try:
            self._inline_modeling_code(checkpoint_folder)
            logger.info(f"✓ Successfully inlined custom modeling code")
        except Exception as e:
            logger.warning(f"Failed to inline modeling code (checkpoint will still work, but may need local code): {e}")
            import traceback
            traceback.print_exc()
        
        # Copy tokenizer files
        try:
            self._copy_tokenizer_files(checkpoint_folder, args)
            logger.info(f"✓ Successfully copied tokenizer files")
        except Exception as e:
            logger.warning(f"Failed to copy tokenizer files (may need to load tokenizer separately): {e}")
            import traceback
            traceback.print_exc()
        
        # Push to Hub
        logger.info("Manually pushing checkpoint to Hub...")
        self._push_checkpoint_to_hub(
            checkpoint_folder, 
            args,
            commit_message=f"Training checkpoint at step {state.global_step}"
        )
        self.last_pushed_checkpoint = checkpoint_folder
        return control
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of a training step. We use this to check if Hub push is about to happen
        and ensure consolidation is complete.
        """
        # Only run on main process
        if not args.should_save:
            return control
        
        # Check if we're about to push (save_steps interval)
        if state.global_step % args.save_steps == 0:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            
            # If this checkpoint exists, verify key files before push
            if os.path.exists(checkpoint_folder):
                logger.info(f"[Pre-push verification] Checking checkpoint before Hub push: {checkpoint_folder}")
                
                import glob
                
                # Verify key files exist
                checkpoint_path = Path(checkpoint_folder)
                config_json = checkpoint_path / "config.json"
                modeling_armt = checkpoint_path / "modeling_armt.py"
                
                # Check for model files (single or sharded)
                pytorch_model = checkpoint_path / "pytorch_model.bin"
                index_file = checkpoint_path / "pytorch_model.bin.index.json"
                shard_pattern = str(checkpoint_path / "pytorch_model-*.bin")
                shard_files = glob.glob(shard_pattern)
                
                model_present = pytorch_model.exists() or (index_file.exists() and len(shard_files) > 0)
                
                all_files_present = all([
                    model_present,
                    config_json.exists(),
                    modeling_armt.exists()
                ])
                
                if all_files_present:
                    if pytorch_model.exists():
                        logger.info(f"[Pre-push verification] ✓ All required files present (single-file checkpoint)")
                    else:
                        logger.info(f"[Pre-push verification] ✓ All required files present (sharded checkpoint, {len(shard_files)} shards)")
                else:
                    logger.error(f"[Pre-push verification] ✗ Missing files!")
                    logger.error(f"  model files: {model_present} (single: {pytorch_model.exists()}, sharded: {index_file.exists() and len(shard_files) > 0})")
                    logger.error(f"  config.json: {config_json.exists()}")
                    logger.error(f"  modeling_armt.py: {modeling_armt.exists()}")
        
        return control
    
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
        
        # Read all ARMT modeling files (support outer, inner, mem_params, thinking)
        files_to_inline = {
            "utils.py": modeling_dir / "utils.py",
            "act_utils.py": modeling_dir / "act_utils.py",
            "language_modeling.py": modeling_dir / "language_modeling.py",
            "model.py": modeling_dir / "model.py",
            "inner_loop_old.py": modeling_dir / "inner_loop_old.py",
            "armt_memory_params.py": modeling_dir / "armt_memory_params.py",
            "thinking.py": modeling_dir / "thinking.py",
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
            code = code.replace("from modeling_amt.utils import", "# inlined utils: removed import")
            code = code.replace("from modeling_amt.inner_loop import", "# inlined inner_loop: removed import")
            code = code.replace("from modeling_amt.armt_memory_params import", "# inlined armt_memory_params: removed import")
            code = code.replace("from modeling_amt.thinking import", "# inlined thinking: removed import")
            
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
                    # Default to outer-loop ARMT
                    model_class = "ARMTForCausalLM"
                    logger.info(f"Using default model class: {model_class}")
            
            # Update config with auto_map
            config["architectures"] = [model_class]
            # Select correct Config class for the given model class
            if model_class in ("ARMTForCausalLM", "InnerLoopARMTForCausalLM"):
                cfg_class = "ARMTConfig"
            elif model_class == "MemoryParamsARMTForCausalLM":
                cfg_class = "MemParamsARMTConfig"
            elif model_class == "ThinkingARMTForCausalLM":
                cfg_class = "ThinkingARMTConfig"
            else:
                # Fallback to ARMTConfig
                cfg_class = "ARMTConfig"
                logger.warning(f"Unknown model class '{model_class}', defaulting AutoConfig to {cfg_class}")

            config["auto_map"] = {
                "AutoConfig": f"modeling_armt.{cfg_class}",
                "AutoModelForCausalLM": f"modeling_armt.{model_class}",
            }
            
            # Save updated config
            config_path.write_text(json.dumps(config, indent=2))
            logger.info(f"Updated config.json with auto_map for {model_class}")
        else:
            logger.warning("config.json not found, could not update auto_map")
        
        # Sync filesystem to ensure all modeling code files are written
        try:
            os.sync()
            logger.info("✓ Modeling code files synced to disk")
        except Exception as e:
            logger.warning(f"Could not sync modeling code files (non-critical): {e}")
    
    def _copy_tokenizer_files(self, checkpoint_dir, args: TrainingArguments):
        """
        Copy tokenizer files to the checkpoint directory.
        
        Looks for tokenizer files in:
        1. The output_dir (saved by Trainer)
        2. Falls back to using AutoTokenizer to save from the base model
        """
        checkpoint_path = Path(checkpoint_dir)
        output_dir = Path(args.output_dir)
        
        # Tokenizer files to look for
        tokenizer_files = [
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'vocab.json',  # For some tokenizers
            'merges.txt',  # For BPE tokenizers
            'tokenizer.model',  # For SentencePiece tokenizers
            'added_tokens.json',
        ]
        
        logger.info(f"Looking for tokenizer files to copy to {checkpoint_path.name}...")
        
        # First, try to find tokenizer files in the output_dir
        copied_count = 0
        for filename in tokenizer_files:
            source_file = output_dir / filename
            if source_file.exists():
                dest_file = checkpoint_path / filename
                if not dest_file.exists():  # Don't overwrite if already exists
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"  Copied {filename} from output_dir")
                    copied_count += 1
                else:
                    logger.debug(f"  {filename} already exists in checkpoint")
                    copied_count += 1
        
        if copied_count > 0:
            logger.info(f"Copied {copied_count} tokenizer file(s) from output_dir")
            return
        
        # If no files found in output_dir, try to save tokenizer from the model
        logger.info("No tokenizer files found in output_dir, attempting to save from base model...")
        
        try:
            from transformers import AutoTokenizer
            
            # Try to get the base model name from config
            config_file = checkpoint_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Try to find the base model identifier
                base_model = None
                if '_name_or_path' in config:
                    base_model = config['_name_or_path']
                elif 'base_model_name' in config:
                    base_model = config['base_model_name']
                
                if base_model and base_model != checkpoint_path.name:
                    logger.info(f"Loading tokenizer from base model: {base_model}")
                    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                    tokenizer.save_pretrained(str(checkpoint_path))
                    logger.info("✓ Saved tokenizer from base model")
                    return
        except Exception as e:
            logger.warning(f"Could not save tokenizer from base model: {e}")
        
        logger.warning("No tokenizer files could be copied or saved")
    
    def _push_checkpoint_to_hub(self, checkpoint_dir, args: TrainingArguments, commit_message=None):
        """
        Manually push the consolidated checkpoint to the Hub.
        
        This is necessary because the Trainer's default push mechanism doesn't always
        handle consolidated DeepSpeed checkpoints correctly.
        """
        checkpoint_path = Path(checkpoint_dir)
        
        try:
            # Initialize HF API
            api = HfApi()
            
            # Get the repo ID - try multiple sources
            repo_id = None
            
            # Try hub_model_id first
            if hasattr(args, 'hub_model_id') and args.hub_model_id:
                repo_id = args.hub_model_id
                logger.info(f"Using hub_model_id: {repo_id}")
            # Fall back to deriving from output_dir
            elif args.output_dir:
                # Extract repo name from output_dir (last component)
                output_path = Path(args.output_dir)
                repo_name = output_path.name
                # Prefix with model class if available → {ModelClass}_run_$N
                if self.model_class_name:
                    repo_name = f"{self.model_class_name}_{repo_name}"
                
                # Try to get username from HF token or use 'user'
                try:
                    user_info = api.whoami()
                    username = user_info['name']
                except:
                    username = os.environ.get('HF_USERNAME', 'user')
                    logger.warning(f"Could not get HF username, using: {username}")
                
                repo_id = f"{username}/{repo_name}"
                logger.info(f"Derived repo_id from output_dir and model class: {repo_id}")
            else:
                logger.error("Cannot determine Hub repo ID - hub_model_id not set and output_dir not available")
                logger.error(f"Available args: push_to_hub={args.push_to_hub}, output_dir={args.output_dir}")
                logger.error(f"hub_model_id={getattr(args, 'hub_model_id', 'NOT SET')}")
                return
            
            if not repo_id:
                logger.error("Failed to determine repo_id for Hub push")
                return
            
            # Determine commit message
            if commit_message is None:
                checkpoint_name = checkpoint_path.name
                commit_message = f"Upload consolidated {checkpoint_name}"
            
            logger.info(f"Manually pushing checkpoint to Hub: {repo_id}")
            logger.info(f"Checkpoint directory: {checkpoint_dir}")
            
            # List files to be pushed
            files_to_push = []
            for file in checkpoint_path.iterdir():
                if file.is_file():
                    files_to_push.append(file.name)
            
            logger.info(f"Files to push: {files_to_push}")
            
            # Ensure repo exists
            try:
                hub_token = getattr(args, 'hub_token', None)
                create_repo(repo_id, exist_ok=True, token=hub_token, private=getattr(args, 'hub_private_repo', False))
                logger.info(f"✓ Repository confirmed/created: {repo_id}")
            except Exception as e:
                logger.warning(f"Repo creation/verification issue (may be non-critical): {e}")
            
            # Collect all files to upload
            import glob
            
            files_to_upload = []
            
            # Essential files (always try to upload if they exist)
            essential_files = [
                'config.json', 
                'modeling_armt.py',
                'training_args.bin', 
                'trainer_state.json',
                'generation_config.json',
                # Tokenizer files (various formats)
                'tokenizer.json',
                'tokenizer_config.json',
                'special_tokens_map.json',
                'vocab.json',
                'merges.txt',
                'tokenizer.model',
                'added_tokens.json',
            ]
            
            for filename in essential_files:
                filepath = checkpoint_path / filename
                if filepath.exists():
                    files_to_upload.append(str(filepath))
            
            # Model files - prefer safetensors over .bin format
            safetensors_index = checkpoint_path / "model.safetensors.index.json"
            safetensors_shards = sorted(glob.glob(str(checkpoint_path / "model-*.safetensors")))
            safetensors_single = checkpoint_path / "model.safetensors"
            
            bin_index = checkpoint_path / "pytorch_model.bin.index.json"
            bin_shards = sorted(glob.glob(str(checkpoint_path / "pytorch_model-*.bin")))
            bin_single = checkpoint_path / "pytorch_model.bin"
            
            if safetensors_index.exists() and safetensors_shards:
                logger.info(f"Detected safetensors sharded format: {len(safetensors_shards)} shards + index")
                files_to_upload.append(str(safetensors_index))
                files_to_upload.extend(safetensors_shards)
            elif safetensors_single.exists():
                logger.info("Detected safetensors single-file format")
                files_to_upload.append(str(safetensors_single))
            elif bin_index.exists() and bin_shards:
                logger.info(f"Detected PyTorch .bin sharded format: {len(bin_shards)} shards + index")
                files_to_upload.append(str(bin_index))
                files_to_upload.extend(bin_shards)
            elif bin_single.exists():
                # PyTorch .bin single file format
                logger.info("Detected PyTorch .bin single-file format")
                files_to_upload.append(str(bin_single))
            else:
                logger.warning("No model checkpoint files found to upload!")
            
            # Upload all files
            hub_token = getattr(args, 'hub_token', None)
            uploaded_count = 0
            failed_count = 0
            total_size = 0
            
            logger.info(f"Uploading {len(files_to_upload)} files to {repo_id}...")
            
            for filepath_str in files_to_upload:
                filepath = Path(filepath_str)
                filename = filepath.name
                
                try:
                    file_size = filepath.stat().st_size / (1024**2)  # MB
                    logger.info(f"  Uploading {filename} ({file_size:.2f} MB)...")
                    
                    api.upload_file(
                        path_or_fileobj=str(filepath),
                        path_in_repo=filename,
                        repo_id=repo_id,
                        commit_message=f"{commit_message}",
                        token=hub_token,
                    )
                    
                    logger.info(f"  ✓ Uploaded {filename}")
                    uploaded_count += 1
                    total_size += filepath.stat().st_size
                except Exception as e:
                    logger.error(f"  ✗ Failed to upload {filename}: {e}")
                    failed_count += 1
                    import traceback
                    traceback.print_exc()
            
            logger.info(f"✓ Manual Hub push completed: {uploaded_count}/{len(files_to_upload)} files uploaded ({total_size / (1024**3):.2f} GB)")
            if failed_count > 0:
                logger.error(f"✗ {failed_count} files failed to upload")
            
        except Exception as e:
            logger.error(f"Failed to manually push checkpoint to Hub: {e}")
            import traceback
            traceback.print_exc()


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

