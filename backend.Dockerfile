# Use an official, lightweight Python image
FROM python:3.11-slim

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