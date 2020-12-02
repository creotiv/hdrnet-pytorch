# Deep Bilateral Learning for Real-Time Image Enhancements
Unofficial PyTorch implementation of 'Deep Bilateral Learning for Real-Time Image Enhancement', SIGGRAPH 2017 https://groups.csail.mit.edu/graphics/hdrnet/

Python 3.6

### Dependencies

To install the Python dependencies, run:

    pip install -r requirements.txt
    
## Datasets
    Adobe FiveK - https://data.csail.mit.edu/graphics/fivek/

## Usage
    
To train a model, run the following command:

    python train.py --test-image=./DSC_1177.jpg --dataset=/dataset_path --lr=0.0001
    
To get all train params run:
    
    python train.py -h
    
To test image run:

    python test.py --checkpoint=./ch/ckpt_0_4000.pth --input=./DSC_1177.jpg --output=out.png
    

## Known issues

* Only PointwiseNN implemented currently
* Dataset has no augmentation which making training difficult 
* No raw HDR input

