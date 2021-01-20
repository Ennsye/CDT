@echo off
pyinstaller --name CDT ^
            --windowed ^
            --noconfirm ^
            --hidden-import dynlib ^
            --add-data saved_data;saved_data ^
            --add-data saved_designs;saved_designs ^
            GUI.py
