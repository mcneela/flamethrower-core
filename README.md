# Flamethrower

Flamethrower is a best-in-class deep learning library intended 
for use in the Flamethrower deep learning course. Students learn to build
their own deep learning library, completely from scratch, and
then use it to train their own neural network models.

## Why Build Your Own Deep Learning Library?

This course is different from other courses that teach deep
learning in that it intends to construct the knowledge it
conveys from first principles. This means that I've done my best
to remove as much abstraction in the way in which concepts are
developed as possible. Of course, that's not to say there's no
abstraction involved whatsoever. Teaching any topic outside of
the rudiments relies on some sort of mutual understanding of
prior knowledge between teacher and student, and this course is
no exception. Desired prerequisites include a solid 
understanding of elementary statistics, calculus, linear algebra
fundamentals (at the very least, familiarity with matrix 
algebra), some systems and data structures knowledge, and 
programming experience in a high-level language (Python, R, 
Julia, Java, C#, etc.) though not necessarily in Python itself. 
The difference here is that the requisite knowledge lies outside
of the domain in which we'll be working. Other courses operate 
in what I'd like to call a top-to-middle approach. They 
introduce machine learning within the context of already 
well-established traditions and tools, perhaps by showing how to
create a convolutional neural network using Keras to classify 
images or how to predict the next word in a sentence using RNN 
language models with PyTorch. From there, they may dive a little
deeper, perhaps touching on a point or two of theory as to how 
the underlying mathematics of the models works and maybe even 
writing down a few equations for the student to peruse. And 
that's where the learning stops. All code is provided a-priori
to the students such that they default to following a recipe. 
They copy down what's written onto their own machines, maybe 
tweak a few hyperparameters here or there, but do they really
experiment? Do they try and compare a wide variety of models on
a single task, do they integrate multiple architectures into a 
single ensemble, do they perform cross-validation passes and 
rigorous hyperparameter searches? Likely not. All they learn is 
a formula. Ingest data, throw together a few Tensorflow layers,
choose a loss function, get decent accuracy, rinse and 
repeat. Is this really sufficient? Is someone who has completed
one of these courses truly suited for an entry-level machine learning
role at a top tech company, and can they really apply 
the knowledge they've learned effectively in their job without 
really understanding what's going on under the hood of the 
systems they utilize? Telling a student to call loss.backward() 
or sess.run() once they've constructed their model doesn't teach them the mathematical principles underlying backprop to the 
point in which they could comfortably derive it themselves (which is necessary when developing anything relatively unique). 
What's more, will they be able to debug the CUDA errors that 
inevitably arise when they construct an invalid model of their 
data? Without understanding how our tools work, we can't 
effectively build, and our development remains stymied by these 
very same tools' constraints. Can 
you imagine trying to implement your own custom CUDA kernels 
when you're unfamiliar with issues of numerical instability, 
low-level programming, and other caveats? In this course, we'll 
fix that as this course is built bottom-up. We start learning 
about neural networks by learning how backpropagation really 
works. And we won't derive symbolic backpropagation rules and 
implement them in slow code to develop a model that would never 
scale in the real world. We'll implement our own reverse-mode 
automatic differentiation framework, the way backprop is 
actually implemented in industry, and we'll build out further 
knowledge atop that underlying foundation. We'll implement our 
own CUDA kernels and we'll achieve speed and performance 
comparable to that of the frameworks used by industry giants. 
We'll implement each and every model and kernel on our own, from 
scratch, using best practices and state of the art techniques. 
We'll develop our own data ingestion pipelines and tie them in 
to our library, then visualize our outputs using our own 
plotting tools. We'll get comfortable with every step of the 
deep learning process from start to finish, and we'll build out 
complex models in Computer Vision, NLP, and Reinforcement 
Learning atop this solid, underlying foundation. And when 
something breaks, we won't be left scratching our heads, because 
we'll have built the framework we're running and will be 
intimately familiar with its implementation.