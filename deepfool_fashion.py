import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, device, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: 输入图像：1x1x28x28
       :param net: network
       :device: cuda or cpu
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    # image = image.reshape(1, 1, 28, 28)
    # image = torch.tensor(image)
    image = image.float().to(device)

    f_image = net(image).data.cpu().numpy().flatten()
    # argsort()为从小到大排序，并依此返回索引
    # [::-1]表示反转
    I = (np.array(f_image)).flatten().argsort()[::-1]
    # I为前10个标签信息
    I = I[0:num_classes]
    label = I[0]

    # 扰动初始化
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    pert_image = pert_image.reshape(1, 28, 28)
    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    # 当k_i！=label时，即对抗样本越过了边界，结束循环
    while k_i == label and loop_i < max_iter:
        """"
        通过内部的for循环可以获得x到各分类边界的距离；
        在外部的while循环中，利用所有边界距离中的最小值对x进行更新。
        重复这一过程，直到的分类标签发生变化。
        """
    # 初始梯度
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)
            # 第k个分类器，对x的梯度，即wk
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig          # wk
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()  # fk

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())  # l'

            # determine which w_k to use
            # 取最短距离
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # 扰动r_i=最短距离pert*单位法向量
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        # 加上扰动时，为(1+overshoot)
        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)

        x = Variable(pert_image, requires_grad=True)
        fs = net(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image
