import requests
r = requests.get('https://www.scss.tcd.ie/doug.leith/CSU44061/2020/reviews_151.jl')
r.encoding = 'utf-8'

with open('data.txt', 'w') as f:
    f.write(r.text)