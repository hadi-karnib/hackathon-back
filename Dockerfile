# Use the official Python image as a base
FROM python:3.12.4

# Set the working directory in the container
WORKDIR /app

# Create a virtual environment in the container
RUN python -m venv /app/.venv

# Ensure the virtual environment is used for all commands
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH to include the /app directory
ENV PYTHONPATH=/app

# Upgrade pip, fastapi, and pydantic
RUN pip install --upgrade pip
RUN pip install --upgrade fastapi pydantic

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the required packages inside the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the FastAPI server using the virtual environment's uvicorn
CMD ["uvicorn", "Model.predict:app", "--host", "0.0.0.0", "--port", "8000"]
