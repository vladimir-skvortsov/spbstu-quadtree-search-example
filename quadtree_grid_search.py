import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def quadtree_grid_search(x, y, C_range, gamma_range, threshold=0.01):
    best_score = 0
    best_params = (0, 0)
    stack = [(C_range, gamma_range)]

    while stack:
        print(f"Stack size: {len(stack)}")
        C_bounds, gamma_bounds = stack.pop()
        C_min, C_max = C_bounds
        gamma_min, gamma_max = gamma_bounds

        # Define the grid
        C_values = np.linspace(C_min, C_max, 3)
        gamma_values = np.linspace(gamma_min, gamma_max, 3)

        for C in C_values:
            for gamma in gamma_values:
                model = SVC(C=C, gamma=gamma)
                score = cross_val_score(model, x, y, cv=5).mean()

                if score > best_score:
                    best_score = score
                    best_params = (2**C, 2**gamma)

        # Subdivide the region if the improvement is significant
        if (C_max - C_min) > threshold and (gamma_max - gamma_min) > threshold:
            mid_C = (C_min + C_max) / 2
            mid_gamma = (gamma_min + gamma_max) / 2
            stack.extend(
                [
                    ((C_min, mid_C), (gamma_min, mid_gamma)),
                    ((C_min, mid_C), (mid_gamma, gamma_max)),
                    ((mid_C, C_max), (gamma_min, mid_gamma)),
                    ((mid_C, C_max), (mid_gamma, gamma_max)),
                ]
            )

    return best_params, best_score
