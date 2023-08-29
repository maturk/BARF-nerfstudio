# BARF-NerfStudio
An unofficial implementation of various BARF algorithms with nerfstudio. This repository includes the following implementations:

1. barf-freq: An implementation of [Vanilla BARF](https://arxiv.org/abs/2104.06405)
2. barf-hash: A modification of the Vanilla BARF method to work with instant-NGP hash grid
3. barf-grad: Gradient scaled hash grid BARF from the paper [Robust Camera Pose Refinement for Multi-Resolution Hash Encoding](https://arxiv.org/abs/2302.01571)

If there ever pops up a new BARF style algorithm, contributions are welcome 😃

# Installation 
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html). Clone the repo and run the following commands:
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
    Author = {Jonathan Hyun Moon, Justin Kerr, and Matias Turkulainen},
    Year = {2023},
    Note = {https://github.com/maturk/BARF-nerfstudio},
    Title = {BARF-nerfstudio: implementation of various BARF algorithms in Nerfstudio}
}
```

## Acknowledgements

The code in this repo was all thanks to the spontaneous collaboration with Jonathan Hyun Moon, Justin Kerr, and Matias Turkulainen.