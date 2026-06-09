%1 -m venv Venv --without-pip
call Venv\Scripts\activate.bat
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
