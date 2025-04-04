@echo off
REM 运行Python脚本整理仓库结构
python organize_repo.py

REM 添加所有文件到Git
echo 添加所有文件到Git...
git add .

REM 提交更改
echo 提交更改...
git commit -m "添加所有项目源代码和目录结构"

REM 推送到GitHub
echo 推送到GitHub...
git push origin master

echo 完成！所有文件已成功推送到GitHub。
pause
