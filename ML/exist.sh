#!/bin/bash
echo "Enter your filename"
read newfile1
if [ -f "$newfile1" ]
then
echo "File is found"
else
   echo "File is not found"
fi