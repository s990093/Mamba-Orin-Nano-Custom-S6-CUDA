# -----------------------------
# Makefile for Mamba inference
# -----------------------------

# 本地模型路徑
MODEL_PATH=model/mamba2-780m

# Prompt
PROMPT="Mamba is a sleek, shadowy creature that moves with unparalleled grace and precision, embodying both silent menace and elegant beauty, its presence commanding the attention of all who dare cross its path."

# -----------------------------
# Run Mamba inference
# -----------------------------
run:
	@.venv/bin/python run_mamba2_metal_inference.py --model $(MODEL_PATH) --prompt $(PROMPT) --device "mps" --max_length 100

# 另一個 target，如果你有不同 script
run_inference:
	@echo "Running run_inference.py with $(MODEL_PATH)..."
	@.venv/bin/python run_inference.py --model $(MODEL_PATH) --prompt $(PROMPT)

# -----------------------------
# Optional: clean cache / logs
# -----------------------------
clean:
	@echo "Cleaning cache and temporary files..."
	@rm -rf __pycache__ *.log *.pt *.safetensors
