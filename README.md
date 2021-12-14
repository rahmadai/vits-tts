# End-to-End VITS TTS

## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. Install espeak first: `apt-get install espeak`
0. Build Monotonic Alignment Search and run preprocessing.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes.
# python preprocess.py --text_index 2 --filelists filelists/backup/wdy_audio_text_train_filelist_5.txt filelists/backup/wdy_audio_text_train_filelist_4.txt filelists/backup/wdy_audio_text_train_filelist_3.txt filelists/backup/wdy_audio_text_train_filelist_2.txt filelists/backup/wdy_audio_text_train_filelist_1.txt

```


## Training
```sh

# Widya Multispeaker
python train_ms.py -c configs/wdy_multispeaker_base.json -m wdy_multispeaker
```
