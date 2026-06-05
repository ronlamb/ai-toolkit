uv init --python 3.12

.\.venv\Scripts\activate

uv pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install huggingface_hub[hf_xet]
uv pip install -U triton-windows
uv pip install -r requirements.txt

