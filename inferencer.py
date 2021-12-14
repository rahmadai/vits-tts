import matplotlib.pyplot as plt
import soundfile as sf

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("/home/server/rahmad/Work/WIN/AI/Speech/vits/model/wdy_multispeaker/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("/home/server/rahmad/Work/WIN/AI/Speech/vits/model/wdy_multispeaker/G_124000.pth", net_g, None)





while(True):
    print("Masukkan Teks :")
    text_input = input("")  # Python 3
    print("Masukkan speaker :")
    speaker_id = input("")
    if(speaker_id == "ayas"):
        speaker_id = 0
    elif(speaker_id == "jean"):
        speaker_id = 1
    elif(speaker_id == "moza"):
        speaker_id = 2
    elif(speaker_id == "davar"):
        speaker_id = 3
    start_time = time.time()
    # stn_tst = get_text("Di sebuah hutan, tinggallah keluarga Bebek di tengah hutan yang terdiri dari Ayah dan empat anak Bebek. Ayah Bebek mempunyai anak bungsu yang lucu dan baik hati, ketiga kakaknya sangat menyanyanginya. Karena sifatnya itulah, ia sangat disayang oleh seluruh binatang hutan.", hps)
    stn_tst = get_text(str(text_input), hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([int(speaker_id)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    end_time = time.time()

    process_time_tts=end_time-start_time
    print(process_time_tts) 
    sf.write("out.wav", audio, 22050, "PCM_16")
