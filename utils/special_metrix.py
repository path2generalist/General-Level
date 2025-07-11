import math
import random

def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _2_sigmoid_minus_1(x):
    return 2 * _sigmoid(x) - 1

def _tanh(x):
    return math.tanh(x)


# mapping param for special metrix
special_metric_dict = {
    # with T
    'MAE': 50,
    'RMS': 50,
    'MSE': 5,
    'RMSE': 5,
    'ABSREL': 0.1,
    'EPE': 1,
    'FID': 25,
    'FVD': 100,
    'FAD': 10,
    'PSNR': 1 / 20,  # higher is better
    'SAD': 10, 
    'RTE': 0.5,
    'CD': 1,
    'MCD': 5,
    # without T
    'WER': None,
    'MS-SSIM': None,
    'MOS': None,
}

HIGHER_IS_BETTER = [
    'PSNR',
]

def map_function_for_special(metrix: str, score: float) -> float:
    """
    Score mapping function for special metrics.
    >>> metrix: metrix name, str, e.g., 'MAE'.
    >>> score: task score, float, e.g., 5.3.
    return: mapped scores, float.
    """
    metrix = metrix.upper()
    T = special_metric_dict[metrix]

    assert score > 0, f'score should be > 0, but found: {score}'
    
    if metrix in HIGHER_IS_BETTER:
        y = _tanh(T * score)
    elif metrix == 'WER':
        y = 1 - score
    elif metrix == 'MS-SSIM':
        y = (score + 1) / 2
    elif metrix == 'MOS':
        y = (score - 1) / 4
    else:  # lower is better
        y = _2_sigmoid_minus_1(T / score)

    return y * 100  # Convert to percentage scale

# • Normalizing WER:
#   y = 1 − x, where x ∈ [0, 1], y ∈ [0, 1].
# • Normalizing MS-SSIM:
#   y = (x + 1) / 2 , where x ∈ [−1, 1], y ∈ [0, 1].
# • Normalizing MOS:
#   y = x − 1 / 4 , where x ∈ [1, 5], y ∈ [0, 1].

if __name__ == '__main__':
    r = random.random()
    print(f"{r = }")
    print(f"{_sigmoid(r) = }")
    print(f"{_2_sigmoid_minus_1(r) = }")
    print(f"{_tanh(r) = }")
    print(f"{_tanh(r / 2) = }")