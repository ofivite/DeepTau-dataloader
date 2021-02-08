# DeepTau-dev

`transform_inputs_numba.ipynb` contains the most up-to-date dev version. The corresponding python script which wraps it up is called `transform_inputs.py`. 

Performance plots has been obtained with:  
```bash
mprof run -o mprof_{batch_size}x{n_batches}_nb.dat transform_inputs.py --path data/ShuffleMergeSpectral_1.root --batch_size {batch_size} --n_batches {n_batches}  
mprof plot mprof_{batch_size}x{n_batches}_nb.dat -o mprof_{batch_size}x{n_batches}_nb.pdf
```
