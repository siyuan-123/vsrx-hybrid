# Third-party notices

VSR-X does not redistribute neural-network weights. Optional adapters may call separately installed projects:

- RapidOCR / PP-OCRv6: follow the downloaded model's license and provenance.
- LaMa: Apache-2.0 upstream code; verify the exact ONNX weight source and hash.
- ProPainter: upstream code and weights are restricted to non-commercial research unless separately authorized.
- STTN: MIT upstream code; verify checkpoint provenance.

Every installed weight must be recorded in `configs/model_manifest.yaml` with an exact SHA-256 hash.
