git add .
@echo off
set /p userInput="Enter commit message: "

git commit -m "%userInput%"

git push origin master

