[![INTERSPEECH 2024 Paper](https://img.shields.io/badge/INterspeech%202024%20Paper-accepted-brightgreen.svg?style=for-the-badge)](https://www.interspeech2024.org/)

![INterspeech Logo](utils\logo-2024.png)
# Missingness-resilient Video-enhanced Multimodal Disfluency Detection

************* This page is under construction. We are working steadily to release the database and codebase at the earliest **********

## Overview

Most existing speech disfluency detection techniques only rely upon acoustic data. In this work, we present a practical multimodal disfluency detection approach that leverages available video data together with audio. We curate an audio-visual dataset and propose a novel fusion technique with unified weight-sharing modality-agnostic encoders to learn the temporal and semantic context. Our resilient design accommodates real-world scenarios where the video modality may sometimes be missing during inference. We also present alternative fusion strategies when both modalities are assured to be complete. In experiments across five disfluency-detection tasks, our unified multimodal approach significantly outperforms Audio-only unimodal methods, yielding an average absolute improvement of 10% (i.e., 10 percentage point increase) when both video and audio modalities are always available, and 7% even when video modality is missing in half of the samples.

![System Diagram](utils\Interspeech_overall.drawio_annoated.png)

## Database

<!-- https://figshare.com/articles/dataset/Audio_Visual_Database_for_Speech_Disfluency/25526953

wget https://figshare.com/ndownloader/articles/25526953/versions/1 -->

## Codebase

### Dependencies
<!-- The required packages are listed in requirements.txt and can be installed as :
```
pip install -r requirements.txt
``` -->


### Training

<!-- Use the following command to train the model from scratch using the preprocessed data which are assumed to be in the folder saved_var/.

```

python training_person_id.py

```

Note : We provide the final version of the best hyperparameters for reproduction.

In the folder data_prep/ additional scripts used for preparing the data to the format in saved_var/ is provided. -->


### Evaluation

<!-- We provide a notebook (evaluation_tutorial_notebook.ipynb) to verify our results on the labeled validation set using the checkpoint available [here](https://drive.google.com/file/d/1444wvkD6kjUjZuhWncTyKaDUcXXO8r0X/view?usp=drive_link). -->



## Citation

<!-- If you use this machine learning codebase or the insights in your research work, please consider citing:
```
@inproceedings{mohapatra2023person,
  title={Person identification with wearable sensing using missing feature encoding and multi-stage modality fusion},
  author={Mohapatra, Payal and Pandey, Akash and Keten, Sinan and Chen, Wei and Zhu, Qi},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2023},
  organization={IEEE}
}
``` -->
