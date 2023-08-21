# Use a lightweight Python image instead of a full Ubuntu image
FROM nvidia/cuda:11.0-base

RUN mkdir /code
WORKDIR /code

ENV PYTHONUNBUFFERED 1

# Install only the necessary packages
RUN apt-get -y update && \
    apt-get -y install curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements_local.txt /code/
RUN pip install --upgrade pip && \
    pip install -r requirements_local.txt
