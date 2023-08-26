# BARF-NerfStudio
An unofficial implementation of various BARF algorithms with nerfstudio. This repository includes the following implementations:

1. An implementation of [Vanilla BARF](https://arxiv.org/abs/2104.06405)
2. A modification of the Vanilla BARF method to work with instant-NGP hash grid
3. Gradient scaled hash grid BARF from the paper [Robust Camera Pose Refinement for Multi-Resolution Hash Encoding](https://arxiv.org/abs/2302.01571)

# Installation 
Clone the repo and run the following commands:
```
conda activate nerfstudio
cd BARF-nerfstudio/
python3 -m pip install --upgrade pip
pip install -e .
ns-install-cli
```

# Running BARF methods
```
ns-train barf-freq --data [DATA]
ns-train barf-hash --data [DATA]
ns-train barf-grad --data [DATA]
```

# Gallery
Photos/results here

# Citation

If you find this work useful, a citation will be appreciated via:

```
@misc{BARF-nerfstudio,
    Author = {authors},
    Year = {2023},
    Note = {https://github.com/maturk/BARF-nerfstudio},
    Title = {BARF-NerfStudio: implementations of coarse-to-fine frequency and hash grid encodings in Nerfstudio}
}
```

## Acknowledgements
