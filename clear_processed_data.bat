@echo off
echo 正在清理 processed_data 目录下的所有文件...
cd /d "C:\Users\UiNCeY\Desktop\emotion\processed_data"
if exist "*.csv" (
    del /f /q "*.csv"
    echo CSV文件已删除
) else (
    echo 没有找到CSV文件
)
if exist "*.txt" (
    del /f /q "*.txt"
    echo TXT文件已删除
)
if exist "*.xlsx" (
    del /f /q "*.xlsx"
    echo XLSX文件已删除
)
if exist "*.xls" (
    del /f /q "*.xls"
    echo XLS文件已删除
)
echo 清理完成！
pause
