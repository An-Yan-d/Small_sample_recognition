# Small_sample_recognition
Small sample computer motherboard recognition.
# Principle
The model is based on the sliding window BOF algorithm.
# Algorithm framework
1. Feature extraction: Use SIFT for feature extraction.
2. Vocabulary construction: K-means is used for clustering. Follow-up tests show that taking 1024 classes works best.
3. Category judgment: After clustering, we get Bag of visual words histograms. Use SVM as discriminative classifier.
4. Generate bounding box:  Reduce the sliding step and window size, and merge the small bounding boxes into larger bounding boxes by taking the merge set method to make the windows more accurate and reasonable in size.
