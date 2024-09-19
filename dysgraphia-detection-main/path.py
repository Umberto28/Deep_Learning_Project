from pathlib import Path

#! change here
BASE = Path('')

#? ML
DEVICE = 'cpu'
CHECKPOINTS = 'checkpoints'

#? IAM dataset
IAM = Path('IAM')
XML = IAM / 'xml'
SETS = IAM / 'SETS'
DATA = IAM / 'DATA'

#? Dysgraphia dataset
DYSG = BASE / 'data'
CSVS = DYSG / 'csv'