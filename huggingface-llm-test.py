import sys
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Increase timeout for Hugging Face downloads (in seconds)
# Default is 10 seconds, increase to 60 seconds for large files
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# HF_TOKEN is automatically detected by huggingface_hub if set as environment variable
# No need to pass it explicitly - the library handles authentication automatically

# Default model - can be overridden via command line argument
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

def get_device():
    """Detect and return the best available device (GPU or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            return device, torch.device(device)
        else:
            print("No GPU detected, using CPU")
            return "cpu", torch.device("cpu")
    except ImportError:
        print("PyTorch not available, using CPU")
        return "cpu", None

def download_model(model_name: str):
    """
    Download the model from Hugging Face.
    
    The huggingface_hub library automatically:
    - Checks if files already exist and skips re-downloading them
    - Uses HF_TOKEN for authentication if available
    - Handles retries with exponential backoff
    - Resumes interrupted downloads
    """
    # Using snapshot_download with repo_id parameter to match detection rule
    # This pattern matches: snapshot_download(repo_id=$MODEL_NAME, ...)
    local_dir = f"./models/{model_name.replace('/', '_')}"
    local_dir_path = Path(local_dir)
    
    # Check if model already exists locally
    if local_dir_path.exists() and any(local_dir_path.iterdir()):
        print(f"Model directory already exists: {local_dir}")
        print("Checking for updates (will skip already downloaded files)...")
    else:
        print(f"Model not found locally, downloading to: {local_dir}")
    
    # Check if token is available (informational)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN detected - using authenticated download")
    else:
        print("No HF_TOKEN found - using anonymous download (may have rate limits)")
    
    # snapshot_download automatically:
    # - Checks if files exist and skips them if they're up-to-date
    # - Only downloads missing or outdated files
    # - Uses the Hugging Face cache system
    print("Downloading/checking model (library skips already downloaded files)...")
    # Downloads always resume when possible (resume_download is deprecated)
    downloaded_dir = snapshot_download(
        repo_id=model_name, 
        local_dir=local_dir,
        local_files_only=False  # Allow checking remote for updates
    )
    print(f"Model ready at: {downloaded_dir}")
    return downloaded_dir

def load_and_use_model(model_dir: str, prompt: str, device_name: str, torch_device):
    """
    Load and use the downloaded model.
    
    Args:
        model_dir: Path to the downloaded model directory
        prompt: Text prompt to use with the model
        device_name: Device name ("cuda" or "cpu")
        torch_device: torch.device object or None
    """
    print(f"Loading model from: {model_dir}")
    print(f"Using device: {device_name}")
    
    model = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model with proper device placement
        if device_name == "cuda" and torch_device is not None:
            # Use device_map="auto" for automatic multi-GPU support, or explicit device
            print(f"Loading model onto GPU: {torch_device}")
            # Show GPU memory before loading
            if torch.cuda.is_available():
                print(f"GPU memory before loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
            
            # Check if GPU compute capability is supported
            # RTX 5090 has sm_120, but PyTorch 2.6 only supports up to sm_90
            try:
                compute_capability = torch.cuda.get_device_capability(0)
                print(f"GPU compute capability: {compute_capability[0]}.{compute_capability[1]}")
                if compute_capability[0] > 9:  # sm_100+ (RTX 5090 is sm_120)
                    print("Warning: GPU compute capability may not be fully supported by PyTorch 2.6")
                    print("Trying GPU load - will fallback to CPU if it fails...")
            except Exception as e:
                print(f"Could not check compute capability: {e}")
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    dtype=torch.float16,
                    device_map="auto"  # Automatically handles device placement
                )
            except RuntimeError as e:
                if "no kernel image is available" in str(e) or "CUDA" in str(e):
                    print(f"\nGPU load failed due to compute capability incompatibility: {e}")
                    print("Falling back to CPU mode...")
                    device_name = "cpu"
                    model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        dtype=torch.float32,
                        device_map=None
                    )
                    model = model.to("cpu")
                else:
                    raise
            # Verify model is on GPU and show memory usage
            if torch.cuda.is_available():
                print(f"GPU memory after loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            if hasattr(model, 'device'):
                print(f"Model loaded on device: {model.device}")
            elif hasattr(model, 'hf_device_map'):
                print(f"Model device map: {model.hf_device_map}")
        else:
            # CPU mode
            print("Loading model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=torch.float32,
                device_map=None
            )
            # Explicitly move to CPU if not already there
            if hasattr(model, 'device') and str(model.device) != 'cpu':
                model = model.to("cpu")
        
        print(f"\nPrompt: {prompt}")
        print("Generating response...")
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        # Note: device_name may have been changed to "cpu" if GPU load failed
        if device_name == "cuda" and torch_device is not None:
            # If using device_map="auto", inputs go to the first device
            # Check model device placement
            model_device = None
            if hasattr(model, 'device') and isinstance(model.device, torch.device):
                model_device = model.device
            elif hasattr(model, 'hf_device_map') and model.hf_device_map:
                # Get first device from device map
                device_map_values = list(model.hf_device_map.values())
                if device_map_values:
                    first_device = device_map_values[0]
                    if isinstance(first_device, int):
                        model_device = torch.device(f"cuda:{first_device}")
                    elif isinstance(first_device, str):
                        model_device = torch.device(first_device)
            
            if model_device:
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                print(f"Inputs moved to device: {model_device}")
            else:
                # Fallback: move to cuda:0
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
                print("Inputs moved to cuda:0")
        else:
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Generate
        print(f"Input device: {inputs['input_ids'].device}")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
        except RuntimeError as e:
            if "no kernel image is available" in str(e) or "CUDA" in str(e):
                print(f"\nGPU generation failed due to compute capability incompatibility: {e}")
                print("Falling back to CPU for generation...")
                # Move model and inputs to CPU
                if device_name == "cuda":
                    model = model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    device_name = "cpu"
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
            else:
                raise
        # Check output device (outputs is a tensor)
        if isinstance(outputs, torch.Tensor):
            print(f"Output device: {outputs.device}")
        
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n" + "="*80)
        print("MODEL RESPONSE:")
        print("="*80)
        print(response)
        print("="*80)
        
    except ImportError as e:
        print(f"Required library not available: {e}")
        print("Install transformers: pip install transformers")
        print("Model files downloaded to:")
        print(f"  {model_dir}")
    except Exception as e:
        print(f"Error loading/using model: {e}")
        import traceback
        traceback.print_exc()
        print(f"Model files are available at: {model_dir}")
    finally:
        # Explicitly release GPU resources
        if device_name == "cuda":
            try:
                import torch
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("\nGPU memory cleared")
            except Exception as cleanup_error:
                print(f"Warning: Error during GPU cleanup: {cleanup_error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and use a Hugging Face LLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python {sys.argv[0]} "What is the capital of France?"
  python {sys.argv[0]} "Explain quantum computing" --model "meta-llama/Llama-3.1-8B-Instruct"
  python {sys.argv[0]} "Hello" --model "Qwen/Qwen2.5-7B-Instruct" --force-cpu
        """
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt to send to the model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    args = parser.parse_args()
    
    # Detect device
    if args.force_cpu:
        device_name, torch_device = "cpu", None
        print("Forcing CPU usage (--force-cpu flag set)")
    else:
        device_name, torch_device = get_device()
    
    # Download the model (this will be detected by the rule)
    # The huggingface_hub library handles retries automatically
    print(f"Downloading model: {args.model}")
    try:
        model_dir = download_model(args.model)
        if model_dir is None:
            print("Error: Model download returned None")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try again later (Hugging Face servers may be busy)")
        print("3. Check if the model name is correct")
        print("4. Ensure HF_TOKEN is set if accessing private/gated models")
        print("5. The library automatically retries with exponential backoff")
        sys.exit(1)
    
    # Load and use the model
    load_and_use_model(model_dir, args.prompt, device_name, torch_device)

