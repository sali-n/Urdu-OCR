# Urdu-OCR

Deep learning project to identify characters of the urdu alphabet.

Created a custom dataset consisting of 1550 images of all 39 alphabets of the urdu language (~40 images per class).

Utilised pytorch to create a VGG-16 and AlexNET models and compared the results between the 2.

Results:

| Model      | Accuracy | F1 Score | Memory (MB) |
|------------|----------|----------|----------|
| VGG - 16    | 96.15%    | 0.964    | 97.86    |
| AlexNet    | 90.71%    | 0.906    | 26.01    |
