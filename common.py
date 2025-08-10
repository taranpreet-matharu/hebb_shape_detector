def activation(x1, w1, x2, w2, threshold):
    return (w1 * x1 + w2 * x2) >= threshold


def activation_matrix(inputs, weights, threshold):
    dot_product = 0
    for i in range(0, len(inputs)):
        dot_product = dot_product + (inputs[i][0] * weights[0][i])
    return dot_product >= threshold


def calculate_output(input, weights, bias_weight):
    output = 0
    dot_product = 0

    for i in range(0, len(input)):
        for j in range(0, len(input)):
            dot_product = dot_product + (input[i][j] * weights[i][j])

    output = dot_product + bias_weight
    return output


def calc_bipolar_activation(output):
    return 1 if output > 0 else -1
