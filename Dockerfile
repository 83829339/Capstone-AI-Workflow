
# Use an official Python runtime as a parent image
FROM python:3.7.5-stretch

RUN apt-get update && apt-get install -y \
python3-dev \
build-essential    
        
# Set the working directory to /app
WORKDIR /capstone

# Copy the current directory contents into the container at /capstone
ADD . /capstone

# Install any needed packages specified in req.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r req.txt

# Define environment variable
ENV NAME CapstoneApp

# Run app.py when the container launches
CMD ["python", "app.py"]
