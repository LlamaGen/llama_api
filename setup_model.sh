mkdir model
wget https://huggingface.co/IlyaGusev/saiga2_7b_ggml/resolve/main/ggml-model-q4_1.bin -P model
pip install -r requirements.txt
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert-llama-ggml-to-gguf.py -i model/ggml-model-q4_1.bin -o model/gguf-model-q4_1.bin
