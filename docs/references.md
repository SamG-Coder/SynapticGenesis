# Research references

SynapticGenesis is an original small C++/CUDA research implementation. These references motivate experiments; they do not establish that its developmental or evolutionary goals already work.

- [Continual Learning Through Synaptic Intelligence, Zenke et al. (2017)](https://arxiv.org/abs/1703.04200): trajectory-based estimates of parameter importance. The optional native penalty is adapted to AdamW and document boundaries, and checked by an independent scalar oracle.
- [Long short-term memory and Learning-to-learn in networks of spiking neurons, Bellec et al. (2018)](https://papers.nips.cc/paper_files/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html): adapting neurons and recurrent connectivity improve temporal computation in their experiments. Our LIF/ALIF cells and controlled cue tests are separate implementations; the paper does not establish that our current language model can use long sentence context.
- [Population Based Training of Neural Networks, Jaderberg et al. (2017)](https://arxiv.org/abs/1711.09846): selecting model states and learning schedules in a population. This is relevant to future selection experiments, not proof of successful two-parent inheritance.
- [Git Re-Basin: Merging Models modulo Permutation Symmetries, Ainsworth et al. (2023)](https://arxiv.org/abs/2209.04836): learned hidden units can be permuted and require alignment before useful merging. Applicability to recurrent spiking state must be tested.
- [Findings of the BabyLM Challenge (2023)](https://aclanthology.org/2023.conll-babylm.1/): small-data language-model evaluation and curriculum experiments. This is a research reference; the BabyLM corpus is not automatically added to our allowlist.
- [Experience Replay for Continual Learning, Rolnick et al. (2019)](https://arxiv.org/abs/1811.11682): bounded memories and replay can limit forgetting in their reinforcement-learning experiments. Our language experiment independently tests whether replayed older source windows help this spiking learner.
- [Dark Experience for General Continual Learning, Buzzega et al. (2020)](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html): combines rehearsal with stored-output distillation. It motivates comparisons that isolate the contribution of memory; stored-logit distillation is not implemented here.
- [Complementary Learning Systems Theory Updated, Kumaran, Hassabis and McClelland (2016)](https://stanford.edu/~jlmcc/papers/KumaranHassabisMcC16CLSUpdate.pdf): discusses interacting fast episodic and slower statistical learning systems, including replay. Our source reservoir and slowly updated parameters are an engineering analogy, not a biological reproduction or proof of human-like development.

The code includes a small adaptive spike quantization codec fixture inspired by the W8ASpike mathematical format. It is a numerical test, not the learner's neuron model or training method. The learner is not a port of W8ASpike or a reproduction of another published spiking language model.

## Selected source editions

- [McGuffey's Eclectic Primer, Revised Edition](https://www.gutenberg.org/ebooks/14642)
- [McGuffey's First Eclectic Reader, Revised Edition](https://www.gutenberg.org/ebooks/14640)
- [McGuffey's Second Eclectic Reader](https://www.gutenberg.org/ebooks/14668)
- [McGuffey's Third Eclectic Reader](https://www.gutenberg.org/ebooks/14766)
- [New National First Reader](https://www.gutenberg.org/ebooks/13853)
- [The Beacon Second Reader](https://www.gutenberg.org/ebooks/15659)
- [The Fairy-Land of Science](https://www.gutenberg.org/ebooks/5726)
- [Home Geography for Primary Grades](https://www.gutenberg.org/ebooks/12228)
