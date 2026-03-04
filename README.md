python -m venv .venv
.venv\Scripts\activate    
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
clear
python main.py

git status
git add .
git commit -m "quick commit"
git push
clear
