# MM-Align: Learning Optimal Transport-based Alignment Dynamics for Fast and Accurate Inference on Missing Modality Sequences

This repository contains the official implementation of the paper: [MM-Align: Learning Optimal Transport-based Alignment Dynamics for Fast and Accurate Inference on Missing Modality Sequences (EMNLP 2022)]()


## Setup

### CMU-MOSI and CMU-MOSEI
Please refer to [this repository](https://github.com/declare-lab/BBFN) to get the .pkl files that store the extracted features (by CMU-MMSDK with integrated COVAREP and P2FA) of the two datasets.

### MELD dataset
You can download the processed dataset (.pkl) from [here](https://drive.google.com/file/d/1RjrYSMpXxg_6r_nUQaysaPyMsldLpMcb/view?usp=sharing).
Alternatively, if you'd like to extract the features by yourself, you can download the raw dataset from [here](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz). Then you can extract the visual and audio features with [ResNet101](https://github.com/v-iashin/video_features) (FPS=25) and [Wave2Vec2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2). Additionally, you need to manually gather text and extracted feature vectors by their IDs and split them into train/dev/test.pkl files.

Next, split the processed dataset into complete/incomplete partitions using `scripts/split_dataset.py`
```bash
python split_dataset.py --data_path <path_to_pickle_files> --seed <seed> --group_id <group_id> --complete_ratio <complete_ratio> --split <split>
```
We provide an example script `script/run_split.sh`, which automatically generates 5 different partitions for a given dataset under the seed 2020-2024.

### Conda Environemnt
```
conda env create -f environment.yml
conda activate mmalign
python -m spacy download en_core_web_sm
```

## Train and Test
```
cd src
python main.py --dataset <dataset_name> --data_path <path_to_dataset> --group_id <group_to_experiment> --modals <modality_pairs> --save_name <name_prefix>
```

The best test results are automatically saved under `results/<save_name>_<modality_pairs>.tsv`

## Citation
Please cite our paper if you find that useful for your research:
```bibtex
@inproceedings{han2022mmalign,
  title={MM-Align: Learning Optimal Transport-based Alignment Dynamics for Fast and Accurate Inference on Missing Modality Sequences},
  author={Han, Wei and Chen, Hui and Kan Min-Yen and Poria, Soujanya},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year={2022}
}
```

## Contact 
Should you have any question, feel free to contact me through [henryhan88888@gmail.com](henryhan88888@gmail.com)

