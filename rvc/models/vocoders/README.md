# pc-NSF-HiFiGAN

The prerequisites downloader automatically stores the pc-NSF-HiFiGAN
checkpoint in this directory with the following name:

`pc_nsf_hifigan_44.1k_hop512_128bin.pth`

The file must contain the `generator` key. The vocoder is shared, loaded only
for inference, and never updated during Mel-VITS training. You can also use
different paths by setting `RVC_PC_NSF_CHECKPOINT` and `RVC_PC_NSF_CONFIG`.
