# 🐯 SynthTIGER: Synthetic Text Image Generator

[![PyPI version](https://img.shields.io/pypi/v/synthtiger)](https://pypi.org/project/synthtiger/)
[![CI](https://github.com/clovaai/synthtiger/actions/workflows/ci.yml/badge.svg)](https://github.com/clovaai/synthtiger/actions/workflows/ci.yml)
[![Docs](https://github.com/clovaai/synthtiger/actions/workflows/docs.yml/badge.svg)](https://github.com/clovaai/synthtiger/actions/workflows/docs.yml)
[![License](https://img.shields.io/github/license/clovaai/synthtiger)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Paper](https://arxiv.org/abs/2107.09313) | [Documentation](https://clovaai.github.io/synthtiger/) | [Datasets](#datasets)

SynthTIGER is synthetic text image generator for OCR model.

<img src="https://user-images.githubusercontent.com/12423224/153699080-29da7908-0662-4435-ba27-dd07c3bbb7f2.png"/>

## Contents

- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Advanced Usage](#advanced-usage)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

## Documentation

The documentation is available at <https://clovaai.github.io/synthtiger/>.

You can check API reference in this documentation.

## Installation

SynthTIGER requires `python>=3.6` and `libraqm`. If you want install dependencies, see [dependencies](depends).

To install SynthTIGER from PyPI:

```bash
$ pip install synthtiger
```

## Usage

```bash
# Set environment variable (for macOS)
$ export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

```
usage: synthtiger [-h] [-o PATH] [-c INTEGER] [-w INTEGER] [-v] SCRIPT NAME [CONFIG]

positional arguments:
  SCRIPT                Script file path.
  NAME                  Template class name.
  CONFIG                Config file path.

optional arguments:
  -h, --help            show this help message and exit
  -o PATH, --output PATH
                        Directory path to save data.
  -c INTEGER, --count INTEGER
                        Number of data. [default: 100]
  -w INTEGER, --worker INTEGER
                        Number of workers. If 0, It generates data in the main process. [default: 0]
  -v, --verbose         Print error messages while generating data.
```

### Examples

#### SynthTIGER text images

```bash
# horizontal
synthtiger -o results -w 4 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config_horizontal.yaml

# vertical
synthtiger -o results -w 4 -v examples/synthtiger/template.py SynthTiger examples/synthtiger/config_vertical.yaml
```

<img src="https://user-images.githubusercontent.com/12423224/153699084-1d5fbb15-0ca0-4a85-9639-6f2c4c1bf9ec.png" width="50%"/>

#### Multiline text images

```bash
synthtiger -o results -w 4 -v examples/multiline/template.py Multiline examples/multiline/config.yaml
```

<img src="https://user-images.githubusercontent.com/12423224/153699088-cdeb3eb3-e117-4959-abf4-8454ad95d886.png" width="75%"/>

## Advanced Usage

### Non-Latin language data generation

<img src="https://user-images.githubusercontent.com/12423224/153699092-c96743d2-9499-4f20-bea1-b7654578b7fa.png" width="40%"/>

1. Prepare corpus

   `txt` format, line by line. ([example](resources/corpus/mjsynth.txt))

2. Prepare fonts

   See [font customization](#font-customization) for more details.

3. Edit corpus path and font path in config file

4. Run gen.py

### Font customization

1. Prepare fonts

   `ttf`/`otf` format. ([example](resources/font))

2. Extract renderable charsets

   ```bash
   python tools/extract_font_charset.py -w 4 fonts/
   ```

   This script extracts renderable charsets for all font files. ([example](resources/font/Ubuntu-Regular.txt))

   Text files are generated in the input path with the same names as the fonts.

3. Edit font path in config file

4. Run gen.py

### Colormap customization

1. Prepare images

   `jpg`/`jpeg`/`png`/`bmp` format.

2. Create colormaps

   ```bash
   python tools/create_colormap.py --max_k 3 -w 4 images/ colormap.txt
   ```

   This script creates colormaps for all image files. ([example](resources/colormap/iiit5k_gray.txt))

3. Edit colormap path in config file

4. Run gen.py

### Template customization

You can implement custom templates by inheriting the base template.

```python
from synthtiger import templates


class MyTemplate(templates.Template):
    def __init__(self, config=None):
        # initialize template.

    def generate(self):
        # generate data.

    def init_save(self, root):
        # initialize something before save.

    def save(self, root, data, idx):
        # save data to specific path.

    def end_save(self, root):
        # finalize something after save.
```

## Datasets

SynthTIGER is available for download at [google drive](https://drive.google.com/drive/folders/1faHxo6gVeUmmFKJf8dxFZf_yRjamUL96?usp=sharing).

Dataset was split into several smaller files. Please download all files and run following command.

```bash
# for Linux, macOS
cat synthtiger_v1.0.zip.* > synthtiger_v1.0.zip

# for Windows
copy /b synthtiger_v1.0.zip.* synthtiger_v1.0.zip
```

**synthtiger_v1.0.zip** (36G) (md5: 5b5365f4fe15de24e403a9256079be70)

- Original paper version.

**synthtiger_v1.1.zip** (38G) (md5: b2757a7e2b5040b14ed64c473533b592)

- Used MJ/ST lexicon instead of MJ/ST label.
- Fixed a bug that applies transformation twice on curved text.
- Fixed a bug that incorrectly converts grayscale to RGB.

| Version | IIIT5k | SVT | IC03 | IC13 | IC15 | SVTP | CUTE80 | Total |
| ------- | ------ | --- | ---- | ---- | ---- | ---- | ------ | ----- |
| 1.0 | 93.2 | 87.3 | 90.5 | 92.9 | 72.1 | 77.7 | 80.6 | 85.9 |
| 1.1 | 93.4 | 87.6 | 91.4 | 93.2 | 73.9 | 77.8 | 80.6 | 86.6 |

### Structure

The structure of the dataset is as follows. The dataset contains 10M images.

```
gt.txt
images/
    0/
        0.jpg
        1.jpg
        ...
        9998.jpg
        9999.jpg
    1/
    ...
    998/
    999/
```

The format of `gt.txt` is as follows. Image path and label are separated by tab. (`<image_path>\t<label>`)

```
images/0/0.jpg	10
images/0/1.jpg	date:
...
images/999/9999998.jpg	STUFFIER
images/999/9999999.jpg	Re:
```

## Citation

```bibtex
@inproceedings{yim2021synthtiger,
  title={SynthTIGER: Synthetic Text Image GEneratoR Towards Better Text Recognition Models},
  author={Yim, Moonbin and Kim, Yoonsik and Cho, Han-Cheol and Park, Sungrae},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={109--124},
  year={2021},
  organization={Springer}
}
```

## License

```
SynthTIGER
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

The following directories and their subdirectories are licensed the same as their origins. Please refer to [NOTICE](NOTICE)

```
docs/
resources/font/
```
