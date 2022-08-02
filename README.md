# Pruned VoiceFixer

This is a pruned version of the original VoiceFixer implementation [here](https://github.com/haoheliu/voicefixer_main).

- [x] Finish initial working version
- [ ] Work on Vocoder - Current candidate: [iSTFTNet](https://github.com/rishikksh20/iSTFTNet-pytorch)


## Dataset Manifest

For this repository, we will be adopting the [Nvidia NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/datasets.html#librispeech)'s manifest format.

```python
# {'input_filepath': <rel. path to noisy audio file>, 'target_filepath': <rel. path to clean audio file>, ... other params}
{"input_filepath": "noisy/0/AuXX152x/0_0.wav", "target_filepath": "clean/0/0_0.wav", "speaker_id": "103-1240", "gender": "f"}
{"input_filepath": "noisy/0/AuXX152x/0_1.wav", "target_filepath": "clean/0/0_1.wav", "speaker_id": "103-1240", "gender": "f"}
```

## Training

1. Update/add the configs in the config folder for your own dataset and experiment
2. Run the following command when running locally:

```bash
python3 train.py
```

## References
```BibTex
 @misc{liu2021voicefixer,   
     title      = {VoiceFixer: Toward General Speech Restoration With Neural Vocoder},   
     author     = {Haohe Liu and Qiuqiang Kong and Qiao Tian and Yan Zhao and DeLiang Wang and Chuanzeng Huang and Yuxuan Wang},  
     year       = {2021},  
     eprint     = {2109.13731},  
     archivePrefix = {arXiv},  
     primaryClass  = {cs.SD}  
 }
```