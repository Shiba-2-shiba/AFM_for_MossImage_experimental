# Sources

The implementation is newly written for the supplied MOSS-Image AFM plan.

- AFM paper: Seunghun Oh and Unsang Park, Attention Frequency Modulation,
  arXiv:2603.28114v1. The mathematical intervention is extended to the
  image-query/text-key partition of MOSS joint attention.
- AFM-for-Krea2-experimental, fe895410558e0272d5f8bbd7c060552e17b1101c:
  architectural reference for joint normalization, chunking, original-output
  preservation and explicit failure accounting. No runtime dependency.
- AFM-for-anima-experimental, 02b2ecc0f794dd94d0d3301e73c87301fb556dec:
  conceptual reference only; no source copied.
- OpenMOSS/MOSS-Image, 52b87fca23acedc399d062d593bec388aee9926a:
  model architecture reference, Apache-2.0.
- ComfyUI 8f04fd6ecf4a548937971e840af2a985529fab76:
  native MOSS, patcher, attention and sampler API contracts.

No third-party model forward implementation is redistributed.
Model files and their licenses are separate from this custom node.
