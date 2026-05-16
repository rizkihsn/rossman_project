# Gunakan Python 3.10 sebagai base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory di container
WORKDIR /code

# Install sistem dependensi yang diperlukan
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt
COPY requirements.txt /code/

# Install python dependensi
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Tambahkan user non-root untuk keamanan (Hugging Face standard)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory ke home user
WORKDIR $HOME/app

# Copy seluruh isi proyek ke container
COPY --chown=user . $HOME/app

# Port standar untuk Hugging Face Spaces adalah 7860
EXPOSE 7860

# Jalankan aplikasi menggunakan Gunicorn
# app.app:app (folder app, file app.py, variable app)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app.app:app"]
