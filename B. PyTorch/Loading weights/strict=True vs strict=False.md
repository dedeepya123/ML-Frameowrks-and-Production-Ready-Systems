strict=True (default)
Model and checkpoint must match exactly.
No missing keys.
No unexpected keys.
Matching tensor shapes.

Best when:

Resuming training.
Loading the exact same architecture.
strict=False
Load all matching parameters and buffers.
Missing keys are left unchanged.
Unexpected keys are ignored.
Matching tensor shapes are still required.

Best when:

Fine-tuning.
Extending a model with new layers.
Loading part of a pretrained model.
