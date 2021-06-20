# DeepTau-dev

This repo contains a proof-of-concept development of a purely pythonic version of `DataLoader` - one of the main modules in the data preparation as a part of [DeepTau](https://github.com/cms-tau-pog/TauMLTools) training. The purpose of the module is to transform jagged arrays of taus and its consitutents (stored in `ROOT TTrees`) onto a 4D "grid" (stored in `numpy` array) with dimensions (tau, eta, phi, feature), where a 1D array (tau, eta, phi, :) corresponds to a single constituent within a tau particle cone. It is built to be independent of [`ROOT`](https://root.cern) and therefore makes use of `uproot + awkward` and also enjoys the speed-up  brought by `numba`. The structure of this repo is the following:

* `notebooks`: "trial and error" code in Jupyter notebooks for testing the approach + visualisation of intermediate steps
*  `configs`: configuration files with descriptive information, to be used by `transform_inputs.py`.
*  `dataloader`: helper functions for performing the transformation
*  `plots`: figures with memory profiling of the code
 
The main script to be run is `transform_inputs.py` which runs over a specified data files (assumed to be "Shuffle & Merge" files) and returns a numpy array as described above. Assuming the ROOT files are in `data` folder it can be run with:

```python
python transform_inputs.py -c configs/input_cfg.yml -d data --n_files 1 --n_batches 10 --batch_size 500
```

Memory/CPU time performance plots can be obtained with (a few of them are shown in `plots` folder):  
```bash
export N_BATCHES=10
export BATCH_SIZE=500
mprof run -o mprof_${BATCH_SIZE}x${N_BATCHES}.dat transform_inputs.py -c configs/input_cfg.yml -d data --n_files 1 --n_batches ${N_BATCHES} --batch_size ${BATCH_SIZE}
mprof plot mprof_${BATCH_SIZE}x${N_BATCHES}.dat -o plots/mprof_${BATCH_SIZE}x${N_BATCHES}.pdf
```
