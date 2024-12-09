# Search Engines and Information Retrieval Readme

## How to Run?
1. Create and Activate Virtual enviroenment 
     - python -m venv .venv 
    - .venv/Scripts/activate
2. Install Requirements
    - pip install -r requirements.txt
3. Run the Program
    - python run.py

## How to change the design 
1. Run the Designer (from terminal)
    - pyqt5_qt5_designer.exe 
    - load ui files and modify it in designer 
    - after modifying save it with .ui extension
2. convert ui to python 
    - pyuic5 dosya_adı.ui -o dosya_adı.py
3. use it in program.
    - You cannot change auto generated py files from .ui so,you should write a wrapper for auto generated py files, which utilizes the auto generated py.
    - whenever the auto generated py files re-generated user changes will be overritten so we need wrapper here. 


> !! You should fallow the pre-created code structure while integrating new pages. 

> © Created and managed by berkay, utilized by berkay and enes.










