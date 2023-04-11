import numpy as np
from sklearn.decomposition import PCA, FastICA
import torch
import numpy as np
import os
from datetime import datetime
import random

def ICA(data):  # data shape [1, 1, length]
    # separate into three parts
    components = 3    # or 5
    data = np.array(data)
    ica = FastICA(n_components=components)
    return ica  # []

def type_align(data):
    # align the data type to torch tensor
    if isinstance(data, torch.Tensor):
        return data.type(torch.float32)
    elif isinstance(data, np.array):
        return torch.from_numpy(data).type(torch.float32)

def save_in_time(result, save_path, *args):
    folder_name = 'output_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = os.path.join(save_path, folder_name)
    os.makedirs(save_folder)

    np.save(os.path.join(save_folder, 'result.npy'), result)

    with open(os.path.join(save_folder, 'flag.txt'), 'w') as f:
        f.write(*args)
    return 0

def save_in_time_hijack(result, ground_truth, save_path, *args):
    folder_name = 'output_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    save_folder = os.path.join(save_path, folder_name)
    os.makedirs(save_folder)

    np.save(os.path.join(save_folder, 'result.npy'), result)
    np.save(os.path.join(save_folder, 'ground_truth.npy'), ground_truth)

    with open(os.path.join(save_folder, 'flag.txt'), 'w') as f:
        f.write(*args)
    return 0


# hijack result evaluation
def evaluate(BSS_data,
             true_data):
    true_data = true_data.squeeze(1)
    sum_err = np.sum((BSS_data - true_data) ** 2)
    return sum_err


def frequency_domain(signal, sample_rate):
    import matplotlib.pyplot as plt

    # Compute the power spectral density using the FFT
    fft_result = np.fft.fft(signal)
    power_spectrum = np.abs(fft_result)**2 / (len(signal) * sample_rate)

    # Compute the corresponding frequencies
    frequencies = np.fft.fftfreq(len(signal), d=1/sample_rate)

    frequencies = frequencies[:int(len(frequencies)/2)]
    power_spectrum = power_spectrum[:int(len(power_spectrum) / 2)]

    return frequencies, power_spectrum
    # # Plot the frequency domain
    # plt.plot(frequencies, power_spectrum)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power')
    
    # # 有可能长度对不上
    # plt.xlim([0, int(sample_rate/2)])
    # plt.show()


def random_select(search_space):
    _lambda = random.choice(search_space)
    return _lambda


from sklearn.manifold import TSNE

def tsne(data, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
    """
    使用t-SNE算法进行数据降维
    :param data: 数据矩阵，每行为一个样本，每列为一个特征
    :param n_components: 降维后的维度数，默认为2
    :param perplexity: t-SNE算法的困惑度，默认为30.0
    :param learning_rate: 学习率，默认为200.0
    :param n_iter: 迭代次数，默认为1000
    :return: 降维后的数据矩阵
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    result = tsne.fit_transform(data)
    # import matplotlib.pyplot as plt
    
    # if plot:
    #     plt.figure()
    #     plt.scatter(result[:, 0], result[:, 1])

    #     plt.show()

    return result