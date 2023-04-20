FROM mcr.microsoft.com/devcontainers/anaconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# Install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends curl gnupg lsb-release bash-completion \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install AzCopy
RUN curl -L --fail -o /tmp/azcopy.tar.gz "https://aka.ms/downloadazcopy-v10-linux" \
  && tar -xvzf /tmp/azcopy.tar.gz -C /tmp \
  && cp /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy \
  && chmod +x /usr/local/bin/azcopy
