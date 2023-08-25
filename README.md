# BARF-NerfStudio
An unofficial implementation of BARF in Nerfstudio including vanilla BARF and hash grid BARF.

# Installation 
Clone the repo and run the following commands:
```
conda activate nerfstudio
cd BARF-NerfStudio/
python3 -m pip install --upgrade pip
pip install -e .
ns-install-cli
```

# Running BARF methods
```
ns-train barf-freq --data [DATA]
ns-train barf-hash --data [DATA]
```

# Gallery
Photos/results here

# Citation

If you find this work useful, a citation will be appreciated via:

```
@misc{BARF-NerfStudio,
    Author = {authors},
    Year = {2023},
    Note = {https://github.com/},
    Title = {BARF-NerfStudio: an implementation of coarse-to-fine frequency and hashgrid encoding in NerfStudio}
}

# Acknowledgements
