import tensorflow as tf
import numpy as np
import pickle
import os
import time
import csv



DELIMITER = ' ||| '

file = open("./ppdb-1.0-s-phrasal", 'rb')
i = 0
writer = open("./sample_phrase_pairs.txt", "w")
for line in file:
    content = ""
        try:
            line_content = line.decode('utf-8').split(DELIMITER)
            print ("Line no: %s " % (i + 1))
            print (line_content[1] + "\t\t\t" + line_content[2])
            i += 1
        except UnicodeDecodeError:
            pass
    i += 1
    if i == 60:
        break
writer.close()
print (i)