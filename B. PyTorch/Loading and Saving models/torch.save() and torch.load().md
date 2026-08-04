torch.save(obj, path)
General-purpose PyTorch serialization utility.
Saves a Python object to disk.
Not specific to models.
torch.load(path)
Reads a saved file.
Reconstructs the original Python object.
Returns the same type of object that was saved.
Responsibilities
API	Responsibility
state_dict()	Extract the model's persistent state (Parameters + Buffers).
load_state_dict()	Copy a saved state into an existing model.
torch.save()	Serialize and write any supported Python object to disk.
torch.load()	Read the serialized object back into Python.
One subtle point

Earlier we simplified state_dict() as returning a dictionary. More precisely, it returns an OrderedDict, which preserves the order of entries. In day-to-day use you can usually think of it as "a dictionary," but it's helpful to know the actual return type because you'll sometimes see it printed or mentioned in documentation.

The complete mental model

Everything we've learned in this module fits into one simple pipeline:

           Training
               │
               ▼
        Model Parameters
               │
               ▼
        model.state_dict()
               │
               ▼
          OrderedDict
               │
               ▼
 torch.save(..., "model.pt")
               │
         ───────────────
            Disk File
         ───────────────
               │
               ▼
     torch.load("model.pt")
               │
               ▼
          OrderedDict
               │
               ▼
 model.load_state_dict(...)
               │
               ▼
      Model Restored

Notice how each API has exactly one responsibility. That separation is one of the reasons PyTorch's design scales so well—from tiny toy models all the way to multi-billion-parameter LLMs
