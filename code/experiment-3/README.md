
## Experiment III:

Experiment was done on two different multimodal dataset configuration: the mnist-fashion
combo and the HIGGS dataset. Each of those dataset consisted of two modes. The experimental
setup was the following,
1. Train an encoder-decoder on each mode
2. Use/fix the encoders trained in (1). Fuse the two modes in a dual-decoder
3. Evaluate on noisy modes
4. Insert EMMA
The objectives are summarized below:
- Compare the results
- Check initialization and end values of the trained parameters (view distributions ...)
- Asses different cooling schedules
- Psychology
