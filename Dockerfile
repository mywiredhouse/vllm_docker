# Option B only: build-from-source fallback. The first-serve path pulls the
# pinned stilldeadcode/vllm-radiance:0.9.3 image instead.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.3
ARG VLLM_REF=v0.9.3

FROM ${ROCM_IMAGE} AS builder
ARG VLLM_REF
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 python3-pip build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone https://github.com/vllm-project/vllm.git && \
    git -C vllm checkout "$VLLM_REF"
# The complete gfx1201/AITER/Triton build is intentionally left as a future
# fallback implementation; this stage is the documented build boundary.
RUN mkdir -p /opt/vllm-build && cp -a /src/vllm /opt/vllm-build/vllm

FROM ${ROCM_IMAGE} AS rocmprune
COPY --from=builder /opt/vllm-build /opt/vllm-build
# Future work: retain only gfx1201 ROCm artifacts and required runtime libs.
RUN mkdir -p /opt/rocm-pruned && cp -a /opt/vllm-build/. /opt/rocm-pruned/

FROM ${ROCM_IMAGE} AS assemble
COPY --from=rocmprune /opt/rocm-pruned /opt/runtime
RUN useradd --create-home --uid 1000 vllm && chown -R vllm:vllm /opt/runtime /home/vllm

FROM ${ROCM_IMAGE} AS final
COPY --from=assemble /opt/runtime /opt/runtime
COPY --from=assemble /etc/passwd /etc/passwd
COPY --from=assemble /etc/group /etc/group
ENV PATH=/opt/runtime/vllm/bin:${PATH} HSA_OVERRIDE_GFX_VERSION=12.0.1
USER 1000
WORKDIR /home/vllm
ENTRYPOINT ["vllm"]
