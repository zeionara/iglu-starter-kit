![IGLU Banner](https://user-images.githubusercontent.com/660004/179000978-29cf4462-4d2b-4623-8418-157449322fda.png)

# **[NeurIPS 2022 - IGLU Challenge](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)** - Starter Kit
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the IGLU Challenge **Starter kit**! It contains:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

Quick Links:

* [The IGLU Challenge - Competition Page](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge)
* [The IGLU Challenge - Slack Workspace](https://join.slack.com/t/igluorg/shared_invite/zt-zzlc1qpy-X6JBgRtwx1w_CBqOV5~jaA&sa=D&sntz=1&usg=AOvVaw33cSaYXeinlMWYC6bGIe33)
* [The IGLU Challenge - Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/iglu-challenge-2022)


# Table of Contents
1. [Intro to the NLP Task](#nlp-task)
2. [Evaluation](#evaluation)
3. [Baselines](#baselines) 
4. [How to test and debug locally](#how-to-test-and-debug-locally)
5. [How to submit](#how-to-submit)
6. [Dataset](#dataset)
7. [Setting up your codebase](#setting-up-your-codebase)
8. [FAQs](#faqs)

# 🙋 NLP Task: Asking Clarifying Questions

This task is about determining when and what clarifying questions to ask. Given the instruction from the Architect (e.g., “Help me build a house.”), the Builder needs to decide whether it has sufficient information to carry out that described task or if further clarification is needed. For instance, the Builder might ask “What material should I use to build the house?” or “Where do you want it?”. In this NLP task, we focus on the research question "what to ask to clarify a given instruction" independently from learning to interact with the 3D environment. The original instruction and its clarification can be used as input for the Builder to guide its progress.

<img src="https://user-images.githubusercontent.com/660004/178754025-1966703c-3e99-4e59-ad79-bf7257e3d35b.png" width="720">

*Top: architect's instruction was clear, not clarifying question gets asked. Bottom: 'leftmost' is ambiguous, so the builder asks a clarifying question.*

# 🖊 Evaluation

Models submitted to the NLP track are going to be evaluated according to both *when to ask* and *what to ask* criteria. The first criterion is a binary classification problem: whether to ask a clarification question or not. Models’ performance are reported using classic metrics such as precision, recall, F1 score, and accuracy. The second criterion evaluates ranking of the list of human-issued clarifying questions. We adopt standard metrics such as MRR.

# Baselines

We have implemented the classification and ranker baseline models. Refer to the following links to learn more about the input, output and using the models.

[`classifiers/random_classifier.py`](classifiers/random_classifier.py) 
[`rankers/random_ranker.py`](rankers/random_ranker.py)

TODO: provide brief description of the baselines.

# How to Test and Debug Locally

The best way to test your models is to run your submission locally.

You can do this naively by simply running  `python local_evaluation.py`. 

# How to Submit

More information on submissions can be found at our [SUBMISSION.md](/docs/submission.md).

#### A high level description of the Challenge Procedure:
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge).
2. **Clone** this repo and start developing your solution.
3. **Train** your models on IGLU, and ensure run.sh will generate rollouts.
4. **Submit** your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com)
for evaluation (full instructions below). The automated evaluation setup
will evaluate the submissions against the IGLU Gridworld environment for a fixed 
number of rollouts to compute and report the metrics on the leaderboard
of the competition.


# 💾 Dataset

Download the public dataset for the NLP Task using the link below, you'll need to accept the rules of the competition to access the data.

https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge-nlp-task/dataset_files

TODO: Add dataset description

# Setting Up Your Codebase

AIcrowd provides great flexibility in the details of your submission!  
Find the answers to FAQs about submission structure below, followed by 
the guide for setting up this starter kit and linking it to the AIcrowd 
GitLab.

## FAQs

* How do I submit a model?
  * More information on submissions can be found at our [SUBMISSION.md](/docs/submission.md). In short, you should push you code to the AIcrowd's gitlab with a specific git tag and the evaluation will be triggered automatically.

### How do I specify my dependencies?

We accept submissions with custom runtimes, so you can choose your 
favorite! The configuration files typically include `requirements.txt` 
(pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the [RUNTIME.md](/docs/RUNTIME.md) file.

### What should my code structure look like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:


```
.
├── aicrowd.json           # Set your username, IMPORTANT: set gpu to true if you need it
├── apt.txt                # Linux packages to be installed inside docker image
├── requirements.txt       # Python packages to be installed
├── local_evaluation.py    # Use this to check your agent evaluation flow locally
├── evaluator/             # Contains helper functions for local evaluation
└── models                 # Place your models and related code here
    ├── rankers            # Folder keep all your ranker code
    ├── classifiers        # Folder keep all your classifier code
    └── user_config.py     # IMPORTANT: Add your classifer and ranker name here
```


Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** See [How do I actually make a submission?](#how-do-i-actually-make-a-submission) below for more details.


### How can I get going with an existing baseline?

See [baselines section](#baselines)

### How can I get going with a completely new model?

Train your model as you like, and when you’re ready to submit, implement the inference class in the `models/classifiers` and `models/rankers` folders. Refer to [`models/README.md`](models/README.md) for a detailed explanation.

Once you are ready, test your implementation `python local_evaluation.py`

### How do I actually make a submission?

First you need to fill in you `aicrowd.json`, to give AIcrowd some info so you can be scored.
The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "neurips-2022-the-iglu-challenge-nlp-task",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
  "gpu": true
}
```

The submission is made by adding everything including the model to git,
tagging the submission with a git tag that starts with `submission-`, and 
pushing to AIcrowd's GitLab. The rest is done for you!

More details are available [docs/submission.md](docs/submission.md).

### Are there any hardware or time constraints?

Your submission will need to complete 1 rollout per task, for ~500 tasks in 7 minutes. 
You may expect that the evaluator will spend most of the time in actions sampling since 
the environment alone can step through all the tasks in around 30 seconds. 
Please, use parallel enviornments to make your agents more time-efficient. 

The machine where the submission will run will have following specifications:
* 1 NVIDIA T4 GPU
* 4 vCPUs
* 16 GB RAM


## Setting Up Details [No Docker]

1. **Add your SSH key** to AIcrowd GitLab

    You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:iglu/neurips-2022-the-iglu-challenge.git
    ```
    
3. **Verify you have dependencies** for the IGLU Gridworld Environment

    IGLU Gridworld requires `python>=3.7` to be installed and available both when building the
    package, and at runtime.
    
4. **Install** competition specific dependencies!

    We advise using a conda environment for this:
    ```bash
    # Optional: Create a conda env
    conda create -n iglu_challenge python=3.8
    conda activate iglu_challenge
    pip install -r requirements.txt
    ```

5. **Run rollouts** with a random agent with `python local_evaluation.py`.


## Setting Up Details [Docker]

Pull the official docker image with installed environment:

```sh
docker pull iglucontest/gridworld_env:latest
```

This image is based on `nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04` base image. If you want to have a custom base,
use the following [Dockerfile](https://github.com/iglu-contest/gridworld/blob/master/docker/Dockerfile)


## Contributors

- Negar Arabzadeh
- Shrestha Mohanty
- [Dipam Chakraborty](https://www.aicrowd.com/participants/dipam)

# 📎 Important links
- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge/problems/neurips-2022-iglu-challenge-nlp-task
- 🗣️ Discussion Forum: https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge/problems/neurips-2022-iglu-challenge-nlp-task/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2022-iglu-challenge/problems/neurips-2022-iglu-challenge-nlp-task/leaderboards

**Best of Luck** 🎉 🎉
