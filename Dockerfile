# syntax=docker/dockerfile:1.2

ARG DEBIAN_FRONTEND=noninteractive

FROM debian:11-slim AS base
ARG TARGETARCH

FROM base AS base_amd64

FROM base AS build-essentials
RUN apt-get update && \
    apt-get install -y build-essential cmake git wget unzip automake libtool pkg-config && \
    rm -rf /var/lib/apt/lists/*

FROM base AS wget
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

FROM base AS python-base
RUN apt-get update && \
    apt-get install -y python3 python3-dev python3-distutils && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py "pip"

FROM python-base AS python-packages
COPY requirements.txt /requirements.txt
COPY requirements-ov.txt /requirements-ov.txt
COPY requirements-wheels.txt /requirements-wheels.txt
COPY requirements-tensorrt.txt /requirements-tensorrt.txt
RUN pip install -r requirements.txt && \
    pip install -r requirements-ov.txt && \
    pip wheel --wheel-dir=/wheels -r requirements-wheels.txt && \
    mkdir -p /trt-wheels && \
    pip wheel --wheel-dir=/trt-wheels -r requirements-tensorrt.txt

FROM build-essentials AS libusb-build
WORKDIR /opt
COPY --from=wget /usr/bin/wget /usr/bin/wget
RUN wget -q https://github.com/libusb/libusb/archive/v1.0.25.zip -O v1.0.25.zip && \
    unzip v1.0.25.zip && cd libusb-1.0.25 && \
    ./bootstrap.sh && \
    ./configure --disable-udev --enable-shared && \
    make -j $(nproc --all) && \
    /bin/mkdir -p '/usr/local/lib' && \
    /bin/bash ../libtool  --mode=install /usr/bin/install -c libusb-1.0.la '/usr/local/lib' && \
    /bin/mkdir -p '/usr/local/include/libusb-1.0' && \
    /usr/bin/install -c -m 644 libusb.h '/usr/local/include/libusb-1.0' && \
    /bin/mkdir -p '/usr/local/lib/pkgconfig' && \
    cd  /opt/libusb-1.0.25/ && \
    /usr/bin/install -c -m 644 libusb-1.0.pc '/usr/local/lib/pkgconfig' && \
    ldconfig

FROM build-essentials AS nginx
COPY docker/build_nginx.sh /deps/build_nginx.sh
RUN --mount=type=tmpfs,target=/tmp --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=bind,source=docker/build_nginx.sh,target=/deps/build_nginx.sh \
    /deps/build_nginx.sh

FROM wget AS go2rtc
WORKDIR /usr/local/go2rtc/bin
RUN wget -qO go2rtc "https://github.com/AlexxIT/go2rtc/releases/download/v1.2.0/go2rtc_linux_${TARGETARCH}" \
    && chmod +x go2rtc

FROM base_amd64 AS ov-converter
COPY --from=python-base /usr/ /usr/
COPY --from=python-packages /usr/local/lib/python3.9/site
