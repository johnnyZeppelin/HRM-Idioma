import numpy as np
import os

folder_path = './data/my-nlp-dataset/train/'
# folder_path = './data/sudoku-extreme-1k-aug-1000/train/'
# folder_path = './data/sudoku-extreme-full/train/'
# folder_path = './data/sudoku-extreme-full/test/'

# folder_path = './data/sudoku-extreme-1k-aug-1000/test/'

file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print(file_list)

for f in file_list:
    if not '.npy' in f:
        continue
    print(f'============\nThe file: {f}\n============')
    test = np.load(os.path.join(folder_path, f))
    print(test)
    print(f'~~~~~~~~~~~~\nThe shape: {test.shape}')

# One case
r = 16  # rank
problem = np.load(os.path.join(folder_path, file_list[2]))
label = np.load(os.path.join(folder_path, file_list[-1]))

print(f'++++++++++++++++++\nThis is case {r}: the problem:')
print(f'{problem[r]}')
print(f'The shape: {problem[r].shape}')

print(f'the solution:')
print(f'{label[r]}')
print(f'The shape: {label[r].shape}')
print(f'++++++++++++++++++')

李斯特 = np.array([2,3,4,5])
print(f'{李斯特}')
print(f'The shape: {李斯特.shape}')

print(len(problem))
print(problem[44223])

# input_new_pro = np.array([
#     1, 6, 1, 1, 7, 1, 5, 1, 1, 1, 1, 2, 5, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 10, 1, 1, 4, 2, 1, 1, 1, 1, 4, 9, 1, 1, 6, 1, 1, 7, 1, 1, 1, 1, 2, 1, 8, 1 , 1, 1, 1, 1, 3, 1, 10, 1, 1, 9, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 8, 1, 1, 1, 6, 1, 1, 2, 1, 1, 0
# ])

# diy_puzz = np.array(
# [0,0,0,0,0,0,8,3,0,0,3,5,4,8,2,1,0,0,0,8,0,0,0,0,0,0,0,0,1,3,2,7,6,0,0,4,8,7,2,0,0,5,0,0,0,6,0,4,0,3,0,2,5,0,9,0,0,0,1,4,0,7,0,0,0,8,0,0,7,0,2,9,3,0,0,0,2,9,5,6,1]
# )

# diy_label = np.array(
#     [4, 6, 9, 7, 5, 1, 8, 3, 2, 7, 3, 5, 4, 8, 2, 1, 9, 6, 2, 8, 1, 6, 9, 3, 7, 4, 5, 5, 1, 3, 2, 7, 6, 9, 8, 4, 8, 7, 2, 9, 4, 5, 6, 1, 3, 6, 9, 4, 1, 3, 8, 2, 5, 7, 9, 2, 6, 5, 1, 4, 3, 7, 8, 1, 5, 8, 3, 6, 7, 4, 2, 9, 3, 4, 7, 8, 2, 9, 5, 6, 1]
# )

# for i in range(len(diy_puzz)):
#     diy_puzz[i] += 1

# print(diy_puzz)
# print(diy_puzz.shape)

# def find(a, b):
#     is_pres = False
#     ind = -1
#     for i in b:
#         ind += 1
#         if np.array_equal(i, a):
#             is_pres = True
#             break
#     if not is_pres:
#         ind = -1
#     return [is_pres, ind]

# print(find(diy_puzz, problem))

# test = np.load('./data/sudoku-extreme-full/test/all__group_indices.npy')

# print(test)

