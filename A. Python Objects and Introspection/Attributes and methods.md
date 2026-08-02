Lesson 4 – Attributes and Methods
Goal

By the end of this lesson, you should be able to answer:

What is an attribute?
What is a method?
Why do they both use .?
Why do some have () and others don't?
How does this relate to ML frameworks?
Let's start with a real example

Imagine you have a car.

The car has:

Color = Red
Speed = 80
Fuel = 50%
Brand = Toyota

These describe the state of the car.

The car can also:

Start()
Stop()
Accelerate()
Brake()

These are actions.

Immediately, you can see two categories:

Information about the object.
Things the object can do.

Python makes exactly the same distinction.

Attributes

An attribute is simply data stored inside an object.

Think of it as a property describing the object's current state.

Example:

class Car:
    pass

Suppose a Car object contains:

brand = "Toyota"
color = "Red"
speed = 80
fuel = 50

These are attributes.

They answer questions like:

What is your brand?
What is your speed?
How much fuel do you have?

They describe the object.

Methods

A method is a function that belongs to an object.

Methods perform actions.

For the same car:

start()

stop()

accelerate()

brake()

Methods change state, compute something, or perform an operation.

A very simple mental model

Think of every object as having two compartments.

Object

├── Information
│      brand
│      speed
│      color
│      fuel
│
└── Actions
       start()
       stop()
       accelerate()

This picture is surprisingly accurate.

Python syntax

Suppose:

car = Car()

Reading information:

car.brand

Notice:

No parentheses.

Because we're reading data.

Calling an action:

car.start()

Notice:

Parentheses.

Because we're asking the object to do something.

Why do both use a dot?

This confuses many beginners.

Why:

car.brand

and

car.start()

Both use ..

The answer is simple.

The dot means:

"Go inside this object and access one of its members."

That member might be:

Data (attribute)
Behavior (method)

The dot itself doesn't care.

Then why do methods need ()?

Because methods are functions.

Imagine you write:

car.start

You are referring to the method itself.

You're not executing it.

Only when you write:

car.start()

does Python actually call the method.

Think of it like this:

car.start

means:

"Show me the instruction manual."

Whereas:

car.start()

means:

"Actually start the car."

Let's connect this to Hugging Face

Suppose:

model = AutoModelForCausalLM.from_pretrained(...)

Now look at:

model.config

Is this data or behavior?

It's data.

It tells us about the model.

No parentheses.

Now:

model.generate(...)

This is behavior.

We're asking the model to perform text generation.

So it needs parentheses.

Another example:

model.device

Information.

model.eval()

Action.

model.training

Information.

model.save_pretrained(...)

Action.

Another PyTorch example

Suppose:

tensor

It might have attributes like:

tensor.shape
tensor.dtype
tensor.device
tensor.ndim

These tell us what the tensor is like.

They don't perform work.

Now methods:

tensor.reshape(...)
tensor.clone()
tensor.cpu()
tensor.cuda()

These perform operations.

Think like an engineer

Whenever you encounter an unfamiliar object, ask yourself:

Is this describing the object?

Then it's probably an attribute.

Examples:

model.config
model.device
model.dtype
tensor.shape
Or is this asking the object to do something?

Then it's probably a method.

Examples:

model.generate()

model.eval()

tensor.reshape()

tokenizer.encode()
Why this matters for inspection

Later we'll use:

dir(model)

You'll see something like:

config
device
dtype
generate
save_pretrained
state_dict
train
eval

Now you'll naturally separate them:

Information:

config

device

dtype

Behavior:

generate()

train()

eval()

save_pretrained()

This makes unfamiliar APIs much less intimidating.

Notes
Topic: Attributes and Methods
What is an Attribute?

An attribute is data stored inside an object. It represents the object's state.

What is a Method?

A method is a function that belongs to an object. It represents behavior.

Why do both use .?

The dot accesses a member of the object. That member may be data (attribute) or behavior (method).

Why do methods use ()?

Methods are functions. Parentheses execute them.

Mental Model

Object = Information + Actions

Attributes describe.

Methods perform.

Key Ideas
Attributes store state.
Methods perform behavior.
. accesses members.
No () → usually data.
() → execute behavior.
Examples
model.config        # Attribute
model.device        # Attribute
model.generate()    # Method
model.eval()        # Method
Where will I see this?
PyTorch tensors
PyTorch models
Hugging Face models
Tokenizers
SDK client objects
Request/response objects
Why does this matter?

When exploring a new object, the first thing you'll do is identify what information it holds and what operations it supports. This helps you understand the object quickly without memorizing its API.
