from pathlib import Path

#? ML
DEVICE = 'cpu'
CHECKPOINTS = 'checkpoints'

#? IAM dataset
IAM = Path('IAM')
XML = IAM / 'xml'
SETS = IAM / 'SETS'
DATA = IAM / 'DATA'

#? Dysgraphia dataset
DYSG = Path('NoAugmentation')
CSVS = DYSG / 'csv'