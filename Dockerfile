FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN python3 -m pip install --upgrade jax[cuda12] jaxlib jaxtyping -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
