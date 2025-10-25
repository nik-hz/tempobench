# ---------- Stage 1: syfco builder ----------
FROM debian:bookworm AS syfco-builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghc cabal-install git libgmp-dev zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build syfco
RUN git clone --depth 1 https://github.com/reactive-systems/syfco.git /src/syfco \
    && cd /src/syfco \
    && cabal update \
    && cabal v2-install --installdir=/out --overwrite-policy=always


# ---------- Stage 2: spot builder ----------
FROM python:3.12.3-bookworm AS spot-builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ make automake libtool pkg-config bison flex swig \
    python3-dev wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# spot version and checksum
ENV SPOT_VERSION=2.14.1
ENV SPOT_SHA256=25df8a6af4e4bb3ae67515ac98e3d37c4303a682e33aaa66e72d74b39459a530

# Build Spot from source
WORKDIR /src
RUN wget http://www.lre.epita.fr/dload/spot/spot-2.14.1.tar.gz \
    && echo "${SPOT_SHA256}  spot-${SPOT_VERSION}.tar.gz" | sha256sum -c - \
    && tar xzf spot-${SPOT_VERSION}.tar.gz \
    && cd spot-${SPOT_VERSION} \
    && ./configure --prefix=/usr/local --enable-python --enable-tools \
    && make -j"$(nproc)" && make install


# ---------- Stage 3: runtime ----------
FROM python:3.12.3-bookworm

# Base tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    mona graphviz ca-certificates zsh curl git gnupg tmux vim tree\
    && rm -rf /var/lib/apt/lists/*


# oh-my-zsh + powerlevel10k
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
    ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# Copy syfco binary from builder
COPY --from=syfco-builder /out/syfco /usr/local/bin/syfco

# Copy Spot binaries + Python bindings
COPY --from=spot-builder /usr/local /usr/local

# Ensure runtime can find libspot
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Python deps (pin pip + deps in one step for reproducibility)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r requirements.txt



WORKDIR /tempo-rl
CMD [ "zsh" ]
