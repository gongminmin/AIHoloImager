%1 -m venv PythonVenv --without-pip
call PythonVenv\Scripts\activate.bat
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
