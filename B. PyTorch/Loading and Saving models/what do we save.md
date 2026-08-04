What changes during training?

Mainly:

Parameters (learnable weights)

Sometimes:

Buffers (persistent state such as BatchNorm statistics)

The architecture and methods remain the same.

Architecture vs State
Architecture → The blueprint (class definition and module structure).
State → The learned numerical values stored in parameters and buffers.
Why save the state?

Because the architecture can usually be recreated from code, while the learned values are the result of training and must be preserved.

The key takeaway

The phrase "save a model" is actually ambiguous.

As an engineer, always think more precisely:

What part of the model do I actually want to preserve?
