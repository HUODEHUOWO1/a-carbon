# Precision Backend Spike

This directory contains non-paper-use spike configs.

- Selected backend: TensorRT (vision) and PyTorch (NLP)
- Scope: Vision (ResNet-50) and NLP (BERT AGNews)
- Measurement contract: `configs/measurement_contract.yaml`

Current files:

- `vision_precision_backend_spike.yaml`
- `vision_imagenette_proxy_profile.yaml`
- `nlp_agnews_precision_spike.yaml`
- `nlp_agnews_precision_spike_intid.yaml`

Notes:

- `paper_use=false` for all spike outputs.
- Capacity lever is disabled in formal matrix; spike focuses on precision-only and runtime viability.

Asset source:

- `E:/a-carbon/experiment_data_assets`



