FROM osrf/ros:humble-desktop-full
ENV USERNAME=ros
ENV DEBIAN_FRONTEND=noninteractive

# Add new user as a sudoer.
RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME && \
        # Replace 1000 with your user/group id
        usermod  --uid 1000 $USERNAME && \
        groupmod --gid 1000 $USERNAME

RUN apt-get update \
   && apt-get -y install --no-install-recommends \
   unzip \
   nmap \
   iputils-ping \
   iproute2 \
   nano \
   git \
   wget \
   curl \
   ca-certificates \
   ffmpeg \
   build-essential \
   cmake \
   pkg-config \
   libopencv-dev \
   python3-dev \
   python3-pip \
   python3-venv \
   python3-scipy \
   python3-matplotlib \
   libjpeg-dev \
   libpng-dev \
   libtiff-dev \
   libatlas-base-dev \
   libgl1-mesa-glx \
   libglib2.0-0 \
   libcanberra-gtk-module

RUN apt-get autoremove -y \
   && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*


# -------------------- Python dependencies (other than ROS) ---------------------
RUN python3 -m pip install --upgrade pip setuptools wheel ultralytics opencv-python-headless \
   pyyaml opencv-contrib-python-headless cython tqdm parrot-olympe==7.5.0 \
   && python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

RUN python3 -m pip install --no-cache-dir setuptools==58.2.0 numpy==1.21.5 qpsolvers[daqp]


# -------------------- ROS setup ---------------------
RUN echo "source /opt/ros/humble/setup.bash" >> /home/ros/.bashrc
RUN echo "source /workspaces/SAFER/install/setup.bash" >> /home/ros/.bashrc

RUN apt-get update && apt-get install -y --no-install-recommends \
   ros-humble-cv-bridge \
   ros-humble-image-transport

# Install gazebo harmonic
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN apt-get update
RUN apt-get install -y gz-harmonic
