import matplotlib.pyplot as plt
import soundfile as sf

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

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

hps = utils.get_hparams_from_file("/home/server/rahmad/Work/WIN/AI/Speech/vits/model/wdy_indo_new_dataset/config.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("/home/server/rahmad/Work/WIN/AI/Speech/vits/model/wdy_indo_new_dataset/G_19000.pth", net_g, None)

stn_tst = get_text("Indonesia merupakan negara kepulauan terbesar di dunia yang memiliki potensi untuk menjadi Poros Maritim Dunia. Poros Maritim Dunia bertujuan menjadikan Indonesia sebagai negara maritim yang besar, kuat, dan makmur melalui pengembalian identitas Indonesia sebagai bangsa maritim, pengamanan kepentingan dan keamanan maritim, memberdayakan potensi maritim untuk mewujudkan pemerataan ekonomi Indonesia.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

sf.write("out.wav", audio, 22050, "PCM_16")