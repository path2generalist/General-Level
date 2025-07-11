<a name="installation"></a>
## 🛠 Installation

### Install with pip
```bash
pip install vbench
pip install -r requirements.txt
```

To evaluate some video generation ability aspects, you need to install [detectron2](https://github.com/facebookresearch/detectron2) via:
   ```
   pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
   ```
    
If there is an error during [detectron2](https://github.com/facebookresearch/detectron2) installation, see [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

### Thrid-party models
Download required pretrained models by following the instructions in [here](pretrained/README.md)



<a name="usage"></a>
## 🚀 Usage
Configure: Modify the model and task type (T2V for text-to-video or I2V for image-to-video) in ``video_generation_evaluate_kit.py``, and then run:
```
python video_generation_evaluate_kit.py
```
This script will automatically:

Generate video outputs 🖥️

Evaluate model performance across metrics 📊