import numpy as np
import random
import copy

# 均值不同的样本
def generate_data(miu1, sigma1, miu2, sigma2, split):
    N = 20000
    s1 = np.random.normal(miu1, sigma1, int(N*split))
    s2 = np.random.normal(miu2, sigma2, int(N * (1-split)))
    s3 = np.concatenate((s1, s2), axis=0)
    return s3


# EM算法
def my_GMM(data):
    new_miu_of_1 = data.max()
    new_sigma_of_1 = 2
    new_miu_of_2 = data.min()
    new_sigma_of_2 = 2

    prob_of_1_all = 0.5
    prob_of_2_all = 1 - prob_of_1_all
    for epoch in range(100):
        prob_of_1_list = []
        prob_of_2_list = []
        for d in data:
            pdf_of_gauss1 = (1 / (np.sqrt(2.0 * np.pi) * new_sigma_of_1)) * \
                            np.exp(-(((d - new_miu_of_1) ** 2) / (2 * new_sigma_of_1 ** 2)))
            pdf_of_gauss2 = (1 / (np.sqrt(2.0 * np.pi) * new_sigma_of_2)) * \
                            np.exp(-(((d - new_miu_of_2) ** 2) / (2 * new_sigma_of_2 ** 2)))

            prob_of_1 = (prob_of_1_all * pdf_of_gauss1) / (pdf_of_gauss1 * prob_of_1_all + prob_of_2_all * pdf_of_gauss2)
            prob_of_2 = 1 - prob_of_1

            prob_of_1_list.append(prob_of_1)
            prob_of_2_list.append(prob_of_2)
        # p(b)
        a = len(data)
        b = np.sum(prob_of_1_list, axis=0)
        prob_of_1_all = np.sum(prob_of_1_list) / len(data)
        # p(a)
        prob_of_2_all = 1 - prob_of_1_all

        # a = np.dot(data, prob_of_1_list)
        # b = np.sum(prob_of_1_list)
        # c = np.sum(prob_of_2_list)
        new_miu_of_1 = np.dot(data, prob_of_1_list) / np.sum(prob_of_1_list)
        new_miu_of_2 = np.dot(data, prob_of_2_list) / np.sum(prob_of_2_list)
        # a = (data - new_miu_of_1) ** 2
        new_sigma_of_1 = np.sqrt(np.dot(prob_of_1_list, (data - new_miu_of_1) ** 2) / np.sum(prob_of_1_list))
        new_sigma_of_2 = np.sqrt(np.dot(prob_of_2_list, (data - new_miu_of_2) ** 2) / np.sum(prob_of_2_list))

        print("gauss1 mean:{:.2f} std:{:.2f}\tgauss2 mean:{:.2f},std:{:.2f}".
              format(new_miu_of_1, new_sigma_of_1, new_miu_of_2, new_sigma_of_2))


if __name__ == '__main__':
    data = generate_data(1, 1, 65, 2, 0.5)
    print("data的均值:{}, 标准差:{}".format(data.mean(), data.std()))
    my_GMM(data)




