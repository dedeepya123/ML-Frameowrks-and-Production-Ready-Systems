Tensor vs Parameter vs Buffer
Tensor
Stores numerical data.
Base building block in PyTorch.
Has no special meaning by itself.
Parameter
A special Tensor.
Represents trainable model weights.
Automatically registered when assigned as an attribute of an nn.Module.
Returned by model.parameters().
Updated by optimizers.
Saved in state_dict().
Buffer
A Tensor that is part of the model's persistent state but is not trainable.
Registered using register_buffer().
Saved in state_dict().
Moved with model.to().
Not returned by model.parameters().
The Golden Mental Model

This is the picture I want you to remember:

                torch.Tensor
                     │
         Stores numerical data
                     │
        ┌────────────┴────────────┐
        │                         │
  nn.Parameter              Buffer (registered)
        │                         │
 Trainable                Persistent State
 Updated by Optimizer      Not Updated
 Appears in parameters()   Not in parameters()
 Saved                     Saved

Notice something elegant:

PyTorch doesn't have three unrelated concepts.

It has one fundamental object (Tensor) and then adds meaning to it depending on its role in the model.

This is the roadmap I propose next

Now that we understand:

Module
Registration
Parameters
Buffers
