import tensorflow as tf
import numpy as np
import pickle
import os
import time
import csv



DELIMITER = ' +++$+++ '

file = open("./quora_duplicate_questions.tsv", 'rb')
reader = csv.DictReader(file, delimiter='\t')
i = 0
writer = open("./processed_quora/sentence_pairs.txt", "w")
for row in reader:
    content = ""
    if row["is_duplicate"] == "1":
        i +=1
    #     content += "Question 1: \n"
        content += row["question1"]
        content += DELIMITER
    #     content += "\n"
    #     content += "Question 2: \n"
        content += row["question2"]
        content += "\n"
        content += "\n"
        content += "\n"
        writer.write(content)
    if i == 100:
        break
writer.close()
print i
