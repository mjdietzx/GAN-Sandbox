# GAN-Sandbox
Standard GAN implemented on top of keras/tensorflow enabling rapid experimentation & research. Branches correspond to stable implementations of GAN architectures (i.e. ACGan, InfoGAN) and other promising variations of GANs (i.e. [GAN hacks](https://github.com/soumith/ganhacks.git), local adversarial loss, etc...).

## Guidelines
The `master` branch serves as a simple, clean and robust starting point for GAN R&D. Contributions are encouraged in the form of new branches and/or improvements to `master`. Ideally branches will follow `master's` coding style and deviate as little (realistically) as possible from it.

## Branches
`master`: Standard GAN.  
`ac-gan`: Auxiliary classifier GAN as described in: [Conditional image synthesis with auxiliary classifier GANs](https://arxiv.org/pdf/1610.09585.pdf).
`info-gan`: Information maximizing GAN as described in: [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657v1.pdf)

Note: `ACGAN` is a more limited form of `InfoGAN`. `InfoGAN` can take an arbitrary number of categorical and continuous latent variables as input to the generator. ACGAN is an InfoGAN in the case where the generator takes one categorical latent variable as input corresponding to the label of the image to be generated.


#### About Waya.ai
Waya.ai is a company whose vision is a world where medical conditions are addressed early on, in their infancy. This approach will shift the health-care industry from a constant fire-fight against symptoms to a preventative approach where root causes are addressed and fixed. Our first step to realize this vision is easy, accurate and available diagnosis. Our current focus is concussion diagnosis, recovery tracking & brain health monitoring. Please get in contact with me if this resonates with you!
