# Use the official Python image as a base
FROM python:3.12.4

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
