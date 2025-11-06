from sklearn.metrics import f1_score, recall_score

def macro_f1(y, p): return f1_score(y, p, average='macro')

def uar(y, p): return recall_score(y, p, average='macro')
