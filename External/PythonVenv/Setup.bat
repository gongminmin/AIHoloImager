Python -m venv PythonVenv
call PythonVenv\Scripts\activate.bat
python -m pip install --upgrade pip
REM install onnxruntime_gpu independently to enable the GPU feature
pip install numpy==1.26.4
pip install onnxruntime-gpu==1.19.0
pip install -r requirements.txt
