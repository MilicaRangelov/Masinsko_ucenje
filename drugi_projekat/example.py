from scipy import spatial
import pandas as pd

df = pd.read_csv('unique-categories.sorted-by-count.embeddings.csv')

v1 = df[df['keyword'] == 'Spaghetti'].values[0][:-1]
v2 = df[df['keyword'] == 'Women\'s Fashion'].values[0][:-1]
result = 1 - spatial.distance.cosine(v1, v2)
print(
    f'Cosine similarity between "Spaghetti" and "Women\'s Fashion": {result}')


v1 = df[df['keyword'] == 'Fashion'].values[0][:-1]
v2 = df[df['keyword'] == 'Women\'s Fashion'].values[0][:-1]
result = 1 - spatial.distance.cosine(v1, v2)
print(
    f'Cosine similarity between "Fashion" and "Women\'s Fashion": {result}')

v1 = df[df['keyword'] == 'biker'].values[0][:-1]
v2 = df[df['keyword'] == 'Women'].values[0][:-1]
result = 1 - spatial.distance.cosine(v1, v2)
print(
    f'Cosine similarity between "biker" and "Women": {result}')
