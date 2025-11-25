run:
	python run_mamba2_mlx.py \
	--model model/mamba2-1.3b-hf/model.safetensors \
	--tokenizer model/mamba2-780m/tokenizer.json \
	--prompt "Mamba is a mysterious" \
	--max_tokens 150 \
	--temperature 0.8 \
	--top_k 50 \
	--top_p 0.92 \
	--repetition_penalty 1.2

echo 'NEXT_PUBLIC_API_URL=http://localhost:8000' > frontend/.env.local



uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
