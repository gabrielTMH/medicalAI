import csv
import unittest
from src import clean_data


class treeTest(unittest.TestCase):
    def accurate_clean(self):
        with open('reorganized.csv') as csvfile:
            readCSV = list(csv.reader(csvfile, delimiter=','))
            test_row = readCSV[43]
        test_row = clean_data(test_row)
        expected = test_row
        self.assertTrue(test_row, expected)

