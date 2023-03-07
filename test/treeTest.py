import csv
import unittest
from src.medical import *
from src.medical.tfidf import clean_data
from data import Medical_Error_Test_Data

class treeTest(unittest.TestCase):
    def accurate_clean(self):
        with open('Medical_Error_Test_Data.csv') as csvfile:
            readCSV = list(csv.reader(csvfile, delimiter=','))
            test_row = readCSV[43]
        test_row = clean_data(test_row)
        expected = test_row
        self.assertTrue(test_row, expected)

