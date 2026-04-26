files = ['analyzer.html', 'finbert.html', 'evaluation.html']
path = 'D:\\sentimentEdge\\hf-sentimentedge\\backend\\templates\\'

for f in files:
    fp = path + f
    c = open(fp, 'r', encoding='utf-8').read()
    c = c.replace('const API=""', 'const API="https://vectorxx-sentiment.hf.space"')
    c = c.replace("const API=''", 'const API="https://vectorxx-sentiment.hf.space"')
    c = c.replace('/api/status', 'https://vectorxx-sentiment.hf.space/api/status')
    c = c.replace('/api/metrics', 'https://vectorxx-sentiment.hf.space/api/metrics')
    c = c.replace('/api/predict', 'https://vectorxx-sentiment.hf.space/api/predict')
    c = c.replace('/api/analyze', 'https://vectorxx-sentiment.hf.space/api/analyze')
    open(fp, 'w', encoding='utf-8').write(c)
    print(f + ' done!')