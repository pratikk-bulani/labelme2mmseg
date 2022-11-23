# labelme2mmseg

## Requirements
```
pip install numpy opencv-python
```

## Execution steps

Please use the following the in the given format:
```
python ./labelme2mmseg.py -i <path consisting of all the json files> -o <path where the output masks should be kept> -l <path to the label file>
```

The first line of the label file NEED NOT be: ```_background_```
The remaining lines should be the labels like:
```
car
horse
...
```
