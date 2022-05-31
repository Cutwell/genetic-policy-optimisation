# Testing

## Concept testing
| | |
|:---:|:---:|
|
| `demo.ipynb` | Complete code in Jupyer notebook |
| `agent-infection-layers.py` | Concept for two-layered cellular automata (agent movement and airflow) |
| `agent-random-walk.py` | Single layer cellular automata with random walking agents |
| `component-map-generator.py` |  |
| `airflow-dynamics.py` | Cellular automata to simulate airflow between windows |

## Custom matrices

* Custom matrices can be generated from images.
* Image is parsed 1-1 into an environment matrix.

### Customisation
1. Image must be 8bit Greyscale (8Bpc GREY) PNG image.
2. Incrementing greyscale values indicate map objects.

| Pixel value | Representation |
|:--:|:--:|
|#000000|Air|
|#010101|Wall|
|#020202|Window|