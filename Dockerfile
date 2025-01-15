# Use a lightweight Python 3.6+ base image with TensorFlow 1.3.0
FROM python:3.6-slim

# Install TensorFlow, Scikit-learn, and required dependencies
RUN pip install tensorflow==1.3.0 \
    networkx==1.11 \
    scikit-learn

# Clean up any default files in the /notebooks directory
RUN rm -rf /notebooks/*

# Copy the current directory to /notebooks in the container
COPY . /notebooks

# Set the working directory to /notebooks
WORKDIR /notebooks

# Default command to start a Bash shell
CMD ["bash"]

