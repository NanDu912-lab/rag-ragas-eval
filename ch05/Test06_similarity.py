import numpy as np

def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == '__main__':
    a = np.array([0.2, 0.8])
    b = np.array([0.3, 0.7])
    c = np.array([0.8, 0.2])
    print(f'{a}, {b} - 余弦相似度：{cosine_similarity(a, b)}')
    print(f'{b}, {c} - 余弦相似度：{cosine_similarity(b, c)}')
