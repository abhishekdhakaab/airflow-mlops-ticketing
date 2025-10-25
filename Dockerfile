FROM apache/airflow:2.9.3

# Copy and install Python deps
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# CPU-only torch wheels (needed for sentence-transformers)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio