# Irani_Chai_Megathon
Frameworks: PyTorch, OpenCV, Face_Detection

# Installation
Navigate to `source code` directory and install the environment either using conda or requirements.txt followed by pip install command.

### conda 
```
conda env create --file conda-gpu.yml
```

### pip
```
pip3 install -r requirements.txt
```

# How to run
After installation or activating the enviroment, execute the following command to run over live input stream:
```
python3 object_tracker.py --video 0 --output $path_to_output_video
```

To run over existing clip, run the following command:
```
python3 object_tracker.py --video $path_to_input_video --output $path_to_output_video
```

Finally, `.csv` and `.txt` files are generated within the source_code directory after completion of the above command.

# Outputs
Link to google drive for model generated output videos, text and csv files are [here](https://drive.google.com/drive/folders/1AKzAMrSUDOyPx-cEn1SsSW_3m7ginJpc).

### Sample Output 
![First image](https://github.com/ar-ohi-srivastav/Irani_Chai_Megathon/blob/main/images/1.png?raw=true)

![Second image](https://github.com/ar-ohi-srivastav/Irani_Chai_Megathon/blob/main/images/2.png?raw=true)

![Thrid image](https://github.com/ar-ohi-srivastav/Irani_Chai_Megathon/blob/main/images/3.png?raw=true)
