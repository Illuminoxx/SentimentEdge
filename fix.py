import os

files = ['analyzer.html', 'finbert.html', 'evaluation.html']
path = 'D:\\sentimentEdge\\hf-sentimentedge\\backend\\templates\\'

for f in files:
    fp = path + f
    c = open(fp, 'r', encoding='utf-8').read()
    c = c.replace('http://localhost:5000', 'https://vectorxx-sentiment.hf.space')
    open(fp, 'w', encoding='utf-8').write(c)
    print(f + ' done!')