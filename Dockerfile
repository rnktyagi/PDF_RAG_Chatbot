# Use an official, lightweight Python image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# (We use standard pip inside the container because it's universally stable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose the port FastAPI runs on
# Change this
EXPOSE 7860

# And change this
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]