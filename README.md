# Encode Attend Navigate

## Overview

[Image Brain]

Tensorflow implementation of "Learning Heuristics for the TSP by Policy Gradient" [Michel Deudon, Pierre Cournut, Alexandre Lacoste, Yossiri Adulyasak, Louis-Martin Rousseau].

## Requirements

- [Python 3.5]()
- [TensorFlow 1.3.0+](https://www.tensorflow.org/install/)
- [tqdm](https://pypi.python.org/pypi/tqdm)

## Usage

- To train a (2D TSP20) model from scratch (data is generated on the fly):
```
> python main.py --max_length=20 --inference_mode=False --restore_model=False --save_to=20/model --log_dir=summary/20/repo
```

NB: Just make sure ./save/20/model exists (create folder otherwise)

- To visualize training on tensorboard:
```
> tensorboard --logdir=summary/20/repo
```

- To test a trained model:
```
> python main.py --max_length=20 --inference_mode=True --restore_model=True --restore_from=20/model
```

## What is Combinatorial Optimization ?

[Comic TSP]

* Combinatorial Optimization: A topic that consists of finding an optimal object from a finite set of objects.
* Sequencing problems: The best order for performing a set of tasks must be determined.
* Applications: Manufacturing, routing, astrology, genetics...

Can we learn data-driven heuristics competitive with existing man-engineered heuristic ?

## What is Deep Reinforcement Learning ?

[Markow Decision Process]

* Reinforcement Learning: A general purpose framework for Decision Making in a scenario where a learner actively interacts with an environment to achieve a certain goal.
* Deep Learning: A general purpose framework for Representation Learning
* Successful applications: Playing games, navigating worlds, controlling physical systems and interacting with users.

## Related Work

Our work draws inspiration from [Neural Combinatorial Optimization with Reinforcement Learning](http://arxiv.org/abs/1611.09940) to solve the Euclidean TSP. Our framework gets a 5x speedup compared to the original framework, while achieving similar results in terms of optimality.

## Architecture

Following [Bello & al., 2016], our Neural Network overall parameterizes a stochastic policy over city permutations. Our model is trained by Policy Gradient ([Reinforce](https://link.springer.com/article/10.1007/BF00992696), 1992) to learn to assign high probability to "good tours", and low probability to "undesirable tours".

### Neural Encoder

[Encoder]

Our neural encoder takes inspiration from advances in Neural Machine Translation (cite self attentive...)
The purpose of our encoder is to obtain a representation for each action (city) given its context.

consists in a RNN or self attentive encoder-decoder with an attention module connecting the decoder to the encoder (via a "pointer"). 

### Neural Decoder

[Decoder]

Similar to [Bello & al., 2016], our Neural Decoder uses a Pointer (cite paper) to effectively point to a city given a trajectory. Our model however explicity forgets after K steps, dispensing with LSTM networks.

### Local Search
We use a simple 2-OPT post-processing to clean best sampled tours during test time.
One contribution we would like to emphasize here is that simple heuristics can be used in conjunction with Deep Reinforcement Learning, shedding light on interesting hybridization between Artificial Intelligence (AI) & Operations Research (OR).

## Results

[quantitative]

[qualitative]

We evaluate on TSP100 our model pre-trained on TSP50 and the results show that that it performs relatively well even though the model was not trained directly on the same instance size as in [Bello & al, 2016]. We believe that the Markov assumption (see Decoder) helps generalizing the model.

## Acknowledgments
[add links]

Ecole Polytechnique (l'X), Polytechnique Montreal and CIRRELT for financial & logistic support
Element AI for hosting weekly meetings
Compute Canada & Télécom Paris-Tech for computational resources.

Special thanks (sorted by name)
Pr. Alessandro Lazaric (SequeL Team, INRIA Lille)
Dr. Alexandre Lacoste (Element AI)
Pr. Claudia D'Ambrosio (CNRS, LIX)
Diane Bernier
Dr. Khalid Laaziri
Pr. Leo Liberti (CNRS, LIX)
Pr. Louis-Martin Rousseau (Polytechnique Montreal)
Magdalena Fuentes (Télécom Paris-Tech)
Mehdi Taobane
Pierre Cournut (Ecole Polytechnique)
Pr. Yossiri Adulyasak (HEC Montreal)


## Author
Michel Deudon / [@mdeudon](https://github.com/MichelDeudon)
