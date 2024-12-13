# EchoDiffNet: Echo-Based Depth Estimation with Wave Conditioned Diffusion Models

This repository contains snippets of test code for [EchoDiffusion] used to demonstrate and validate the methods described in the paper. While the full research code is not released to protect project integrity and sensitive information, we provide this test code to verify the model's accuracy.

## Dataset

* **Replica-Dataset:** Available from the [Replica](https://github.com/facebookresearch/VisualEchoes) repository. 
* **Matterport-Dataset:** An extension of the existing [Matterport3D](https://niessner.github.io/Matterport/) dataset.

## Evaluation

Configure the relevant YAML files before testing. Pre-trained model parameters can be [downloaded](https://drive.google.com/file/d/15MLo6jRcxtDE-xNHwRy5lpVAwz1pBCAY/view?usp=drive_link).

To run the test script:
```
pip install -r requirements.txt
python test.py
```

