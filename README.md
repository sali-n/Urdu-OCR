# Urdu-OCR

Deep learning project to identify characters of the urdu alphabet.
Created a custom dataset consisting of 1550 images of all 39 alphabets of the urdu language.
Utilised pytorch to create a standard VGG-16 and AlexNET and compared the results between the 2.

Results:

| Model      | Accuracy | F1 Score | Memory (MB) |
|------------|----------|----------|----------|
| VGG - 16    | 99.04%    | 0.989    | 179.9    |
| AlexNet    | 92.31%    | 0.922    | 96.76    |
