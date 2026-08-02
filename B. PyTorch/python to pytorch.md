Mental Model: Python → PyTorch
Step 1: Everything in Python is an object

Python is an object-oriented language.

Everything you create is an object.

Examples:

x = 10          # int object
name = "Alice"  # str object
nums = [1,2,3]  # list object
d = {"a": 1}    # dict object

Every value is an object of some class.

type(10)         -> int
type("abc")      -> str
type([1,2])      -> list

So the first thing to remember is:

Python programs are made up of objects.

Step 2: PyTorch builds on Python

PyTorch is not a new language.

It is a Python library.

That means it introduces new classes (new object types).

Just like Python has:

int
str
list
dict

PyTorch provides classes such as:

torch.Tensor
torch.nn.Module
torch.nn.Parameter
torch.optim.Optimizer
torch.device

So now your program contains both Python objects and PyTorch objects.

Python Objects

int
str
list
dict
tuple

        +

PyTorch Objects

Tensor
Module
Parameter
Optimizer
Device

Notice that PyTorch isn't replacing Python—it's extending it.

Step 3: nn.Module is just another Python class

This is the key realization.

class MyModel(nn.Module):
    ...

nn.Module is simply a Python class written by the PyTorch developers.

Its purpose is:

To represent a neural network or any reusable neural network component.

Examples:

nn.Linear
nn.Conv2d
nn.LayerNorm
nn.Embedding
Your own MyModel

All of these are subclasses of nn.Module.

So whenever you create:

model = MyModel()

model is just another Python object.

The difference is:

Its class (nn.Module) gives it neural-network-specific capabilities.

Step 4: Why doesn't PyTorch use plain Python objects?

Suppose we wrote:

class MyModel:

    def __init__(self):
        self.weight = torch.randn(10, 10)

This is a perfectly valid Python class.

But PyTorch has no idea:

Which tensors are trainable?
Which tensors should be saved?
Which tensors should move to the GPU?
Which tensors belong to child layers?

So PyTorch introduces a smarter base class:

class MyModel(nn.Module):

Now nn.Module can automatically manage all of those things.

Step 5: Another special object — nn.Parameter

This is where your earlier question fits perfectly.

Suppose we write:

self.weight = nn.Parameter(...)

nn.Parameter is another PyTorch class.

When nn.Module sees:

self.weight = nn.Parameter(...)

it says:

"Ah! This object is a Parameter. I should register it as a trainable parameter."

But if we write:

self.hidden_size = 4096

that's just an int.

PyTorch says:

"This is ordinary Python data. I'll just store it as a normal attribute."

The Complete Decision Tree

Whenever you assign something inside an nn.Module, imagine PyTorch asking:

self.some_attribute = value

            │
            ▼
What type of object is "value"?

            │
            ├──────────────► nn.Parameter
            │                   │
            │                   ▼
            │          Register as Parameter
            │
            ├──────────────► nn.Module
            │                   │
            │                   ▼
            │          Register as Submodule
            │
            ├──────────────► Buffer (registered via register_buffer)
            │                   │
            │                   ▼
            │          Register as Buffer
            │
            └──────────────► Anything else
                                │
                                ▼
                     Normal Python Attribute

Notice something important:

PyTorch doesn't care about the attribute name.

It cares about the type of the object you're assigning.

This is the mindset I want you to have

When you see code like:

self.embedding = nn.Embedding(...)
self.linear = nn.Linear(...)
self.hidden_size = 4096
self.dropout = nn.Dropout(...)
self.scale = 0.125

Don't just read the names.

Your brain should automatically classify them:

embedding  → nn.Module
linear     → nn.Module
dropout    → nn.Module

hidden_size → int (normal attribute)
scale       → float (normal attribute)

Later, if you see:

self.weight = nn.Parameter(...)

you immediately know:

Parameter
↓
Trainable
↓
Appears in parameters()
↓
Saved in state_dict()
↓
Moved by model.to(device)
Notes (Reference)
Python → PyTorch Mental Model
Everything in Python is an object.
PyTorch is a Python library that introduces new object types.
nn.Module is a Python class representing a neural network component.
nn.Parameter is a special tensor type that nn.Module automatically registers as trainable.
torch.Tensor stores numerical data.
nn.Module manages the model's structure and trainable state.
The golden rule

PyTorch decides how to treat an attribute based on the type of the object assigned to it—not the name of the attribute.
