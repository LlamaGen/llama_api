FROM python:3.11

WORKDIR "/app"
RUN mkdir "model"
RUN wget https://huggingface.co/IlyaGusev/saiga2_7b_gguf/resolve/main/ggml-model-q4_K.gguf -P model
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]