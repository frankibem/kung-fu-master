# CS 5331 - Pattern Recognition Project
The goal of this project was to train an agent to play Kung Fu Master using reinforcement learning. Report can be found [here](https://github.com/frankibem/kung-fu-master/blob/master/kung%20fu%20master.pdf).

## Dependencies
* Python 3.5
* numpy
* [CNTK](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/)
* [OpenAI Gym](https://gym.openai.com/docs) (with Atari package)
* [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) (to visualize training progress)

Experiments were run on an Ubuntu 16.04 virtual machine with 7GB RAM and 2 cores running on Microsoft Azure. We encountered difficulties when trying to install gym[Atari] on a machine running windows 10.

## Training
To see the available arguments, run:
> python3 atari_train.py -h

To train using the default options (same as in report), run:
> python3 atari_train.py KungFuMaster-v0

The trained model and logs (as well as checkpoints) will be saved to <cur_dir>/chkpt

To visualize the training progress, run:
> tensoroard --logdir=chkpt/logs

__Although any atari environment can be specified, the preprocessing step assumes that the current environment is for Kung Fu Master. You may have to modify wrapper.py to remove those details.__

## Evaluation
To see the available arguments, run:
> python3 atari_eval.py -h

To obtain a baseline average using a random agent, run:
> python3 atari_eval.py path -rnd -r

The -r flag turns on rendering so you can see the play but slows down the evaluation significantly. You can set the number of episodes using the '-e' flag.

To evaluate using the final trained model (you can turn off rendering), run:
> python3 atari_eval.py chkpt/KungFuMaster-v0.dqn -r

If training was interrupted before completion, you can still evaluate with one of the saved checkpoints:
> python3 atari_eval.py chkpt/KungFuMaster-v0_<number>.dqn -c -r

Watch the random agent [here](https://www.youtube.com/watch?v=oxN2fm0-YWA) and the trained agent [here](https://www.youtube.com/watch?v=Fsr8bSn7Mzk)
