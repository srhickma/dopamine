# Space Invaders with Reinforcement Learning

This is the Group 2 research project for CS486 at the University of Waterloo. It is built on top of
Google's [Dopamine](https://github.com/google/dopamine) research framework.
Our goal is to expand upon the Dopamine framework to provide an agent more closely resembling the
Rainbow([Hessel et al., 2018][rainbow]) paper, with the following 4 extensions implemented:

* Noisy Networks([Fortunato et al., 2018][noisy_net])
* N-step Bellman Updates([Mnih et al., 2016][a3c]),
* Prioritized Experience Replay([Schaul et al., 2015][prioritized_replay])
* Distributional Reinforcement Learning([C51; Bellemare et al., 2017][c51])

The original Dopamine implementations of Deep Q-Learning
([Mnih et al., 2015][dqn]), Rainbow([Hessel et al., 2018][rainbow]), and
Implicit Quantile Networks([Dabney et al., 2018][iqn]) are included for comparison.

Our goal is to have an easily reproducible way to compare the results of adding Noisy Networks on top
of the previous Dopamine implementation. Our pre-processing thus follows the same process given by
[Machado et al. (2018)][machado].

## Instructions

### Setup
This project requires Python 3.6 and above.

To run experiments, the project needs to be downloaded via git clone.

It is optional, though recommended, to create a virtual environment to isolate the libraries required for
this project. To create a virtual environment and activate it within the directory the project was
cloned to:

```
python3 -m venv ./env
source env/bin/activate
```

To setup the environment and install this project's dependencies:

```
pip install -U pip
pip install -r requirements.txt
```

### Training agents

The entry point to the standard Atari 2600 experiment is
[`dopamine/discrete_domains/train.py`](https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/train.py).
To train the DRA:
```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/dra_run \
  --gin_files dopamine/agents/rainbow/configs/rainbow.gin
```

To train the Noisy DRA:
```
python -um dopamine.discrete_domains.train \
  --base_dir /tmp/noisy_dra_run \
  --gin_files dopamine/agents/rainbow/configs/noisy_rainbow.gin
```

By default, this will kick off an experiment lasting 200 million frames.
The command-line interface will output statistics about the latest training
episode:

```
[...]
I0824 17:13:33.078342 140196395337472 tf_logging.py:115] gamma: 0.990000
I0824 17:13:33.795608 140196395337472 tf_logging.py:115] Beginning training...
Steps executed: 5903 Episode length: 1203 Return: -19.
```

To get finer-grained information about the process,
the experiment parameters in the gin file can be adjusted.
This is useful to inspect log files or checkpoints, which
are generated at the end of each iteration.

More generally, the whole project is easily configured using the
[gin configuration framework](https://github.com/google/gin-config).

To continue training from a previous checkpoint, run the same command as to begin
training, but with the same base_dir. The checkpoint will automatically be loaded
and the agent will continue from where it left off.

To visualize the returns as an agent trains, run:
```
tensorboard --logdir {base_dir}
```

### Running the experiment
To run the 50 game experiment on the 200th iteration's checkpoint of an agent with base directory `/tmp/noisy_dra_run`:
```
python3 experiment.py \
    --root_dir results \
    --restore_checkpoint /tmp/noisy_dra_run/checkpoints/tf_ckpt-199
```
This will create a `results` folder in the current directory with a video of the game with the highest score.

The 7 number summary and human normalized scores are printed to stdout as an array once the experiment completes:
```
Performance metrics: [[1.17500000e+03 8.75625000e+03 1.63375000e+04 2.39187500e+04
  3.15000000e+04 1.63375000e+04 1.51625000e+04 1.07642952e+01]]
```
The performance metrics, in order, are: min, 25th percentile, 50th percentile, 75th percentile, max, mean,
standard deviation, and median human normalized score.

## References

[Bellemare et al., *The Arcade Learning Environment: An evaluation platform for
general agents*. Journal of Artificial Intelligence Research, 2013.][ale]

[Machado et al., *Revisiting the Arcade Learning Environment: Evaluation
Protocols and Open Problems for General Agents*, Journal of Artificial
Intelligence Research, 2018.][machado]

[Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning*.
Proceedings of the AAAI Conference on Artificial Intelligence, 2018.][rainbow]

[Mnih et al., *Human-level Control through Deep Reinforcement Learning*. Nature,
2015.][dqn]

[Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*. Proceedings
of the International Conference on Machine Learning, 2016.][a3c]

[Schaul et al., *Prioritized Experience Replay*. Proceedings of the International
Conference on Learning Representations, 2016.][prioritized_replay]

[Fortunato et al., *Noisy Networks For Exploration*, Proceedings of the International
Conference on Learning Representations, 2018.][noisy_net]

[machado]: https://jair.org/index.php/jair/article/view/11182
[ale]: https://jair.org/index.php/jair/article/view/10819
[dqn]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[a3c]: http://proceedings.mlr.press/v48/mniha16.html
[prioritized_replay]: https://arxiv.org/abs/1511.05952
[c51]: http://proceedings.mlr.press/v70/bellemare17a.html
[rainbow]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17204/16680
[iqn]: https://arxiv.org/abs/1806.06923
[dopamine_paper]: https://arxiv.org/abs/1812.06110
[noisy_net]: https://openreview.net/forum?id=rywHCPkAW
