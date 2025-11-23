import argparse
from mamba_metal_model import MambaModelMetal

def main():
    parser = argparse.ArgumentParser(description="Run Mamba 130M Inference on Metal")
    parser.add_argument("--prompt", type=str, default="Mamba is a type of snake", help="Input prompt")
    parser.add_argument("--max_length", type=int, default=50, help="Max generation length")
    args = parser.parse_args()

    print("Initializing Mamba Model on Metal...")
    model = MambaModelMetal("state-spaces/mamba-130m-hf")
    
    print(f"\nStarting generation for prompt: '{args.prompt}'")
    output = model.generate(args.prompt, max_length=args.max_length)
    
    print("\nFinal Output:")
    print(output)

if __name__ == "__main__":
    main()
