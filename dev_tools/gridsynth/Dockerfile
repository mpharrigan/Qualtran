FROM ubuntu:24.04
RUN apt-get update -q && apt-get upgrade -yq
RUN apt-get install -yq build-essential curl libffi-dev libffi8 libgmp-dev libgmp10 libncurses-dev pkg-config
RUN curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | \
    BOOTSTRAP_HASKELL_NONINTERACTIVE=1 \
    BOOTSTRAP_HASKELL_INSTALL_STACK=1 \
    BOOTSTRAP_HASKELL_INSTALL_HLS=1 \
    BOOTSTRAP_HASKELL_ADJUST_BASHRC=N \
    sh
RUN . /root/.ghcup/env && cabal install newsynth
ENTRYPOINT ["/root/.cabal/bin/gridsynth"]

# Usage:
# 1. First, you need to build the container (this will take several minutes).
#
#   cd [path to the directory that contains this DOCKERFILE]
#   docker build -t gridsynth .
#
# 2. Now, you can run it. Any arguments will be passed directly to gridsynth
#
#   docker run -t gridsynth [args]
#
