# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:04:04 2025

@author: ychuang
"""

from heapq import merge

def merge_sorted_lists(x_lists, y_lists):
    # Combine all x and y pairs
    combined = []
    for x_list, y_list in zip(x_lists, y_lists):
        combined.extend(zip(x_list, y_list))
    
    # Sort the combined list based on x values
    combined.sort(key=lambda pair: pair[0])
    
    # Unzip into final x and y lists
    final_x, final_y = zip(*combined)
    
    return list(final_x), list(final_y)

# Example usage with 5 data sets
x1 = [1, 3, 5]
y1 = [10, 30, 50]

x2 = [2, 4, 6]
y2 = [20, 40, 60]

x3 = [0, 7]
y3 = [5, 70]

x4 = [1.5, 3.5]
y4 = [15, 35]

x5 = [2.5, 4.5]
y5 = [25, 45]

x_lists = [x1, x2, x3, x4, x5]
y_lists = [y1, y2, y3, y4, y5]

final_x, final_y = merge_sorted_lists(x_lists, y_lists)

print("Final x:", final_x)
print("Final y:", final_y)
