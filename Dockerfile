FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN mkdir /code
WORKDIR /code

ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install tzdata and preconfigure the timezone
RUN apt-get update && apt-get install -y tzdata
RUN echo $TZ > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements_local.txt /code/
RUN pip install -r requirements_local.txt
