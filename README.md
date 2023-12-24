# Hello SSM

This repo will contain a set of 'HelloWorld' style projects to help understand SSM-related architectures

### `hippo/`

This is a minimal implementation of HIPPO.

It comes from an ongoing series of talks by BeeGass on [sap.ient.ai](http://sap.ient.ai/) Discord server

Basically Bee ripped out a near-minimal machinery from his repo at https://github.com/BeeGass/HiPPO-Jax to get a demo.

The .ipynb trains on TinyShakespeare

On a GPU it's able to manage 10 epochs in 5 minutes and spit out very wholesome looking Shakespearean prose, despite the model only containing some 250 weights!