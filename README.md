# Network Intrusion Detection System Machine Learning Project - COMP SCI 642: Introduction to Information Security - Fall 2024 - Prof. McDaniel

## Due date: See Canvas


## Description

In this project, you have a collection of network data, both benign and
malicious. The goal is to train a variety of machine learning models that are
capable of predicting if future network traffic is benign or malicious. Your
models will then be "deployed" by a test program and observe additional, unseen
traffic to see how well they perform.

This project includes both coding/testing models and writing responses/attaching
plots to questions.

Please follow the instructions carefully and turn in the results as directed
before the deadline above.

## Dependencies and cloning instructions

- Install [Docker](https://docs.docker.com/engine/install/) and
  [git](https://git-scm.com/downloads) on your machine. If you are a macOS user,
  you should also authorize Docker to access your files, see this
  [guide](https://support.apple.com/en-gb/guide/mac-help/mchld5a35146/mac).

- Configure git on your machine and optionally [add a SSH key to your GitHub
  account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh):
    ```sh
    git config --global user.name "Bucky Badger"
    git config --global user.email uw-bucky-badger@wisc.edu
    ```

- Accept the GitHub Classroom link provided on the Canvas Assignment, a private
  GitHub repository is created for you, clone it on your machine (you can find
  the HTTPS or SSH url by clicking on the green button named *Code*):

    `git clone
    <HTTPS_OR_SSH_URL>UW-Madison-COMPSCI642/nids-ml-<YOUR_GITHUB_USERNAME>.git`

- A `Dockerfile` is provided under `.devcontainer/` (for direct integration with
VS Code). Using VS Code with Docker and VS Code Dev Containers extension as
described on [this
guide](https://gist.github.com/yohhaan/b492e165b77a84d9f8299038d21ae2c9) will
likely be the easiest for you. If you have issues with sharing the git
credentials with your Docker container, refer to this
[documentation](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials).

- In case, you would like to manually build the image and deploy the Docker
container to test your code (if you are not using VS Code but another
development workflow), follow the instructions below:

  1. Build the Docker image (needs to be done only once normally):
    ```sh
    docker build --platform linux/amd64 -t cs642-projectnids-docker-image .devcontainer/
    ```

  2. Every time you want to test your code and if you have exited the container
     you had previously created, you will have to deploy a new Docker container:
    ```sh
    docker run --platform linux/amd64 --rm -it -v ${PWD}:/workspace/projectnids \
        -v ${HOME}/.gitconfig:/home/vscode/.gitconfig \
        -v ${HOME}/.ssh:/home/vscode/.ssh \
        -w /workspace/projectnids \
        --entrypoint bash cs642-projectnids-docker-image:latest
    ```
    Note: you may have to edit the source path for the `.gitconfig` and `.ssh`
    volumes (for now it looks for those in your home directory on your machine).
    These 2 volumes are needed so that your git configurations and potential ssh
    keys are accessible from within the Docker container, respectively.

- You are ready to start on your project! It is highly recommended to keep track
  of your modifications often by committing and pushing your changes to your
  private repository.


## Project Details

### Overview

There are 4 main experiments to write, and 1 additional experiment that is
optional but will provide extra credit if completed. Each experiment corresponds
to a different machine learning model or classification setting, and has a
function provided in `main.py`, which should be filled in with the student code.

All the experiments have the same general structure:
- Takes as input `data` and `labels`, the training examples and corresponding
  labels.
  _Note: The data and labels passed in have already been properly loaded
  and preprocessed, no additional normalization or transformations are
  necessary. However, the data passed in includes all the data you have access
  to, so splitting the data is encouraged._
- Define and fit your model. Model definition is already present in the starter
  code, but the arguments to each model should be adjusted.
- Compute some metrics, generate some plots.
- Return your trained model. _Note: Model saving is already handled for you._

The only difference between experiments is in the types of labels that are
passed in, the model types you should use, and the metrics you should compute.

### Helpful functions and important notes

The `sklearn` package that should be used for defining and training your models
already comes with a suite of *very* useful functions and attributes that will
greatly help you, particularly with the required plots, printing metrics, and
doing data exploration. This assignment is not designed to require many lines of
intense coding, but rather to teach you how to design and train machine learning
models, interpret results, and think critically about what makes a good model.
You are highly encouraged to read through the scikit-learn
[documentation](https://scikit-learn.org), consult online examples, and use
existing functions from the library when completing this project.

Additionally, the dataset provided and models capable of achieving good accuracy
are small enough that no special equipment (e.g., a GPU) is necessary to
complete the assignment.

### Experiments

In addition to what is described below, make sure to check and complete the
`write-up/` folder for the plots/metrics needed for each experiment, as there
are different requirements for each experiment. You may also want to produce
additional metrics to assess the performance of your models before submitting
them for testing.

1. **Decision Tree, Binary Classification.** You will train a decision tree
   model using the `DecisionTreeClassifier` class from `sklearn.tree`. This
   class accepts a variety of hyperparameter arguments that can be passed to
   change the structure/functionality of the model. You should experiment with a
   variety of different combinations, making note of what you try and what works
   well in the write-up. Since this is binary classification, `labels` contains
   `0` for benign samples and `1` for all malicious samples.

2. **Decision Tree, Multiclass Classification.** This experiment is mostly the
   same as experiment 1, but with multiclass classification. Therefore, `labels`
   now specifies what attack the malicious samples belong to: `0` for benign
   samples, `1` for DoS traffic, `2` for Probe traffic, `3` for R2L traffic, and
   `4` for U2R traffic.

3. **Neural Network, Binary Classification.** You will train a decision tree
   model using the `MLP` class from `sklearn.neural_network`. This class accepts
   a variety of hyperparameter arguments that can be passed to change the
   structure/functionality of the model. You should experiment with a variety of
   different combinations, making note of what you try and what works well in
   the write-up. Since this is binary classification, `labels` contains `0` for
   benign samples and `1` for all malicious samples.

4. **Neural Network, Multiclass Classification.** This experiment is mostly the
   same as experiment 3, but with multiclass classification. Therefore, `labels`
   now specifies what attack the malicious samples belong to: `0` for benign
   samples, `1` for DoS traffic, `2` for Probe traffic, `3` for R2L traffic, and
   `4` for U2R traffic.

5. **EXTRA CREDIT, Pytorch Implementation of experiment 4 model.** This
   experiment is similar to experiment 4, but uses the `torch` library rather
   than `sklearn`. PyTorch does not have an automatic `.fit()` method on its
   models, so a training loop will have to be manually written and some
   additional adjustments have to be made to the data for it to work in PyTorch.
   - No plots/graphs are needed for this portion.
   - No accuracy threshold needs to be met on the test program. As long as the
     training loop works as intended and the saved model is capable of running
     in the test program, full points will be awarded for the coding portion of
     this experiment.


### How to run your code

To execute your program, run `python3 main.py --exp <EXP_NUM>`. This will run
the function corresponding to that experiment number and will save the trained
model to `exp<EXP_NUM>_model.pkl` (or `exp<EXP_NUM>_model.pt` for the extra
credit experiment).

Once you are happy with your model, you can submit it to the test program to be
tested against the hidden test set.

To test your models on the hidden test set, run `./642test <MODEL_FILE_NAME>`. Optionally, for binary classification settings you can also
specify a threshold (default is 0.5) for classifying samples as attacks. In this
case, run `./642test <MODEL_FILE_NAME> <THRESHOLD>` to include the
threshold.

The test program will output a variety of statistics generated from
`sklearn.metrics.classification_report`. Models will be given full points for
the testing portion of the grade once they reach at least 88% accuracy on the
test program.

**NOTE: The number of times the test program is run is tracked. To encourage
thoroughly evaluating your model before deploying it for testing, bonus points
will be awarded for each model (not including the extra credit model) that
reaches the test accuracy threshold within 5 submissions to the test program
(half points if under 10 attempts).**

### Write-up

In addition to coding and testing all the models mentioned above, there are also
a series of questions in `write-up/` that should be answered and turned in along
with your code through your GitHub repository. See [this
guide](https://www.markdownguide.org/basic-syntax/) on formatting Markdown
files.

This folder contains questions that are not specific to any one
function/experiment in `general_questions.md`, as well as questions that are
specific to each experiment in `experiment_questions.md` and require showing the
plots that should be generated by your code. Please ensure that you have filled
in answers for all the sections and attached images of plots when applicable.
Finally, the questions corresponding to the extra credit model (experiment
number 5) are present in `extra_credit_questions.md`. These questions can be
answered by anyone that attempts the extra credit portion of the assignment,
even if it was not finished.


## How to turn in

1. Commit and push your changes to the private repository created by GitHub
   Classroom before the deadline.

2. **And** submit on Canvas before the deadline both your *GitHub username* and
   the *commit ID* that should be graded. Otherwise, you will receive a 0 for
   the assignment. Note that the TA will grade your repository as available at:
   `https://github.com/UW-Madison-COMPSCI642/nids-ml-<YOUR_GITHUB_USERNAME>/commit/<COMMIT_ID_TO_GRADE>`

**Tip:** you can test that the TA will obtain the correct version of your code
and that they will be able to compile it by:

- Cloning again your GitHub repository into *another* location on your machine
  or VM.

    `git clone
    <HTTPS_OR_SSH_URL>UW-Madison-COMPSCI642/nids-ml-<YOUR_GITHUB_USERNAME>.git`

- Checking out to the commit ID you would like to be graded.

    `git checkout <COMMIT_ID_TO_GRADE>`

- Compiling your code and testing that everything works as expected.

    `./642test <MODEL_FILE_NAME> <THRESHOLD>`


## Note

**Like all assignments in this class you are prohibited from copying any content
from the Internet or discussing, sharing ideas, code, configuration, text or
anything else or getting help from anyone in or outside of the class. Consulting
online sources is acceptable, but under no circumstances should *anything* be
copied. Failure to abide by this requirement will result dismissal from the
class as described in the course syllabus on Canvas.**
