# Codename RVC Fork 4 — Mel-VITS + pc-NSF-HiFiGAN

This variant uses a single conversion architecture:

```text
source audio → ContentVec + F0 → speaker-conditioned Mel-VITS
             → 128-bin log-mel → frozen pc-NSF-HiFiGAN → waveform
```

The trainable model contains a content encoder, VAE posterior, normalizing
flow, speaker embedding, and mel decoder. The vocoder is shared by all models,
does not receive gradients, and is not included in voice checkpoints.

## Acoustic format

- sample rate: 44,100 Hz
- FFT/window: 2048
- hop: 512
- mel bins: 128
- frequency range: 40–16,000 Hz
- compression: natural logarithm

ContentVec and F0 are interpolated to the exact mel time grid.

## Dataset preprocessing and storage

The default profile is optimized for Mel-VITS context and disk usage:

- 6-second slices with 0.1 seconds of overlap;
- FLAC PCM24 storage for 44.1 kHz ground-truth audio;
- 384-frame training crops (about 4.46 seconds);
- ContentVec features stored as float16 and restored to float32 when loaded;
- continuous F0 stored as float32 and coarse F0 stored as uint8;
- temporary 16 kHz audio removed only after all feature and F0 files have
  been verified.

The extraction panel provides an option to retain the temporary 16 kHz audio
when repeated feature extraction is required. Running preprocessing again
clears previously generated audio, feature, and F0 directories so stale files
cannot be mixed with a new slicing configuration.
The F0 extraction range is configurable (30–1,600 Hz by default) and is saved
into the experiment config so training uses the same coarse-F0 mapping.

## Linux installation

```bash
chmod +x run-install.bat run-fork.bat
./run-install.bat
```

The `.bat` extension is kept for compatibility with the project layout, but
both files are Bash scripts for Linux.

The installer downloads Miniconda to `miniconda3/` inside the repository and
creates `env/` as a Conda environment with Python 3.10.18. It does not use the
system Python. If it finds a `venv`-based `env/` or one using another Python
version, it preserves it with the `.python-incompatible-<date>` suffix before
creating the correct environment. The launcher runs the application with
`conda run`.

The pc-NSF-HiFiGAN checkpoint is downloaded automatically from RIFT-SVC when
the interface starts. It is stored at
`rvc/models/vocoders/pc_nsf_hifigan_44.1k_hop512_128bin.pth`. The compatible
`config.json` is already in the same directory.

Start the interface:

```bash
./run-fork.bat
```

To enable `torch.compile` on Linux:

```bash
RVC_TORCH_COMPILE=1 ./run-fork.bat
```

You can also set:

- `RVC_PC_NSF_CHECKPOINT`: path to the exported generator;
- `RVC_PC_NSF_CONFIG`: path to the vocoder JSON file;
- `TORCH_INDEX_URL`: PyTorch package index used by the installer;
- `PYTHON_VERSION`: Python version used to create the environment;
- `MINICONDA_VERSION`: Miniconda release to download (`latest` by default).

## Training

Training retains AdamW, AdaBelief, RAdam, Ranger21, Schedule-Free AdamW/RAdam,
warmup, exponential decay, cosine annealing, AMP, TF32, gradient checkpointing,
gradient clipping, two-sample KL, DDP, and checkpoints. It also supports
gradient accumulation, monotonic KL warmup, configurable KL free bits,
speaker-balanced batches, a speaker-stratified validation split, and EMA
weights for validation and export.

For memory-constrained GPUs, EMA can be stored in system RAM and updated with a
decay-adjusted interval. Branchwise waveform training completes the acoustic
backward first, then recomputes only a configurable microbatch for the frozen
pc-NSF/STFT objective. The vocoder and STFT branches also use activation
recomputation, reducing peak VRAM without shortening the 384-frame context.
The SID conversion cycle is a separate branch as well, so its graph is
released before the reconstruction branch is built.

Full-model checkpointing covers every TextEncoder transformer layer, the
PosteriorEncoder WaveNet, each normalizing-flow coupling layer, and the mel
decoder. The attention path uses PyTorch SDPA when available, allowing CUDA to
select FlashAttention or its memory-efficient kernel. Relative-key and
relative-value terms are preserved; relative values are recomputed in query
chunks instead of retaining a full attention-probability tensor.
Training reports peak allocated/reserved CUDA memory in the console and under
the TensorBoard `memory/` group, resetting the peak counters each epoch.

FP16 automatic mixed precision is enabled by default on CUDA for both training
and inference. Model weights remain in FP32 during training, while autocast
uses FP16 for supported kernels and keeps Gaussian sampling, KL, and acoustic
loss accumulation in FP32. Dynamic loss scaling is used to protect gradients.
CPU execution automatically falls back to FP32 without changing the configured
default. The precision selector in the Settings tab is the single source of
truth for the interface; CLI training can override it with `--use_fp16`.

The frozen pc-NSF-HiFiGAN participates in training as a differentiable
renderer: its weights never change, but a multi-resolution waveform STFT loss
can send gradients back to the predicted mel. The interval and rendered frame
count are configurable to control VRAM and training time.

Alternatively, `Use pc-NSF only during validation` keeps the vocoder entirely
out of the training loop. It is loaded lazily for a configurable number of
held-out waveform-metric batches and immediately unloaded afterward. In this
mode Mel-VITS is optimized only by its acoustic, KL, conversion, and speaker
objectives.

The Linux launchers set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default to reduce CUDA
allocator fragmentation. An explicitly supplied environment value is
preserved.

In addition to mel reconstruction, multi-speaker batches can perform SID
swapping and cycle consistency in content space. Gradient-reversal speaker
classification removes identity leakage from content, while decoded-mel
classification reinforces the requested target identity. Continuous log-F0
and an explicit voiced/unvoiced channel complement the coarse F0 embedding.
Optional pitch augmentation shifts the waveform and both F0 representations
together. Multi-speaker-only objectives are automatically harmless for a
single-speaker dataset.

Checkpoints from the previous waveform architectures are incompatible and are
explicitly rejected.

## Hybrid-FSQ acoustic alternative

The training tab offers **Hybrid-FSQ** alongside **Mel-VITS**. Both predict the
same 128-bin, 44.1 kHz / hop-512 mel representation rendered by the bundled
pc-NSF-HiFiGAN. The quality-patched Hybrid-FSQ v1 uses:

- an architecturally coarse deterministic decoder, preserving the need for
  the stochastic global and local paths;
- an 8-dimensional rate-controlled global Gaussian latent constrained to a
  time-constant low-rank DCT correction;
- parallel slow `[8, 8, 8, 8]` and fast `[5, 5, 5, 5]` FSQ residual paths;
- factorized, geometry-aware scalar priors with locally soft posterior
  targets instead of unstable 512/125-way joint labels;
- contextual TCN prior heads and progressive exposure of only the local
  residual decoder to inference-prior values;
- stronger final-mel, multiscale, temporal-delta and frequency-delta
  supervision.

Training is end-to-end in one phase and does not load the waveform vocoder.
Earlier Hybrid-FSQ v1 checkpoints are incompatible with the factorized local
prior; existing mel, feature, F0, statistics and packed caches remain reusable.
The first run creates `hybrid_stats.pt` and `hybrid_manifest.json` in the
experiment folder using only the training split. Exported inference `.pth`
files retain the required normalization and residual-cap tensors while
omitting all three training-only posterior networks.

For large datasets, the first optimized run creates
`hybrid_packed_cache/`: four indexed mmap files for mel, ContentVec, coarse F0
and voiced F0. Packed training shuffles large blocks while keeping each batch
physically sequential; this avoids scattered page faults in caches larger than
RAM without fixing the order between epochs. Individual derived mel caches are
removed only after the packed cache is committed atomically; source audio and
extracted features remain untouched. Subsequent epochs read only the requested
256-frame ranges and never decode source audio. Packed-cache parallelism and
locality can be tuned with `data.packed_loader_workers`,
`data.packed_prefetch_factor` and `data.packed_locality_batches`.

CLI users should select `--vocoder_arch hybrid_fsq` during extraction and
`--architecture Hybrid-FSQ` during training. The UI selects the matching
configuration automatically.

## Stochastic Residual Conformer-GAN acoustic alternative

The training tab also offers **Stochastic-Residual-Conformer-GAN**. It uses a
native PyTorch Conformer-Lite generator for the coarse mel structure and a
small Gaussian global/local residual latent for stochastic texture. A random
area mel discriminator and a voicing-aware mel discriminator are used only
during training; exported checkpoints retain only the generator and its
prior. pc-NSF-HiFiGAN remains a separate frozen renderer for validation and
inference.

The default configuration uses six 192-channel Conformer blocks, an 8-channel
global latent and a 12-channel local latent at quarter mel resolution. The
local prior is exposed progressively during the single training phase so the
decoder sees the same stochastic path used at inference. `noise_scale`-style
controls are available through the model's global/local stochastic scales,
while a nonzero seed keeps conversion reproducible.

CLI users should select `--vocoder_arch stochastic_conformer_gan` during
extraction and `--architecture Stochastic-Residual-Conformer-GAN` during
training. The UI selects the matching configuration automatically.

## Raw-NSF-Waveform-GAN architecture

The training tab also offers **Raw-NSF-Waveform-GAN**, a fourth experimental
architecture that performs waveform-to-waveform conversion without VITS,
Conformer or an external content encoder at inference. A raw waveform encoder
produces a constrained content representation, a separate prosody stem
predicts F0/voicing/energy, and the trainable full NSF-HiFiGAN decoder renders
the target speaker. Multi-period and multi-resolution waveform discriminators
are used during training; exported checkpoints contain only the generator.

The architecture is initialized from the bundled pc-NSF-HiFiGAN decoder when
available. Its raw encoder and new speaker-conditioning layers are trained
from scratch, while the full NSF source path is enabled for the waveform GAN.
Training still uses mel and multiresolution STFT as auxiliary objectives, but
inference accepts only source waveform and target speaker ID (an optional F0
override is retained for the pitch-guidance controls).

## Responsible use

Only use voices and recordings for which you have permission, and clearly
disclose when audio has been synthesized or converted.
