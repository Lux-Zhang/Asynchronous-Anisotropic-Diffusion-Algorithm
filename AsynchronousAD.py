import cv2
import numpy as np


def GaussianKernel2D(Size=5, Sigma=0):
    """
        用来生成二维的高斯核，因为 cv2.GaussianBlur() 并非简单的高斯滤波。
        具体方法是先确定高斯核的大小，使用 cv2.getGaussianKernel() 生成
        两个一样的高斯核。然后kx乘ky的转置。
        参数描述：
            Size  - 高斯核的大小（尺寸）
            Sigma - 高斯核的方差
    """

    # 确定高斯核大小，最小不小于5*5
    Size = int(Size / 2) * 2 + 1
    if Size < 5:
        Size = 5

    # 生成两个一样的一维高斯核，若需要x方向和y方向方差不一样，可生成两个不一样的高斯核
    KernelX = cv2.getGaussianKernel(Size, Sigma)
    KernelY = cv2.getGaussianKernel(Size, Sigma)

    # kx乘ky的转置
    Result = np.multiply(KernelX, np.transpose(KernelY))
    return Result


def CutKernelGenerator(Size, Direction):
    """
        用来生成八个方向的各向异性梯度模的计算核，使用的方法是从中心向四周生长的方法。
        生成的有两种基本核，正方向的（N），斜向45度的（NW），通过旋转来达到八个方向。
        参数描述：
            Size      - 梯度模核的大小（尺寸）
            Direction - 梯度模核的方向
    """

    # 确定梯度模核大小，最小不小于3*3
    Size = int(Size / 2) * 2 + 1
    if Size < 3:
        Size = 3
    Dim = 3

    # 确定核的中心值
    Center = ((Size ** 2) - 1) * 2 / 8

    # 以下分为两个大分值，正向的核和45度的核，该分支为正向的核
    if (Direction == 'N') or (Direction == 'S') or (Direction == 'W') or (Direction == 'E'):

        # 确定中心核（种子）
        CenterKernel = np.array([[0, 2, 0],
                                 [0, -1 * Center, 0],
                                 [0, 0, 0]])

        # 开始生长
        while Dim < Size:

            # 增大中心核
            CenterKernel = np.pad(CenterKernel, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

            # 获得此时中心核的尺寸
            height, width = CenterKernel.shape

            # 分奇偶两种情况赋值，该分支为奇情况
            if (((width - 1) / 2) + 1) % 2 == 0:

                # 求出此时核的周长
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                Slice1 = int(((width - 1) / 2) - ((Factor - 1) / 2))
                Slice2 = int(((width - 1) / 2) + ((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                CenterKernel[0, Slice1:Slice2] = 2

            # 该分支为偶情况
            else:

                # 偶情况要退化成上一次核的的周长
                width = width - 2
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                Slice1 = int(((width - 1) / 2) - ((Factor - 1) / 2))
                Slice2 = int(((width - 1) / 2) + ((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                CenterKernel[0, Slice1 + 1:Slice2 + 1] = 2

                # 核的这些位置赋值为1
                CenterKernel[0, Slice1] = 1
                CenterKernel[0, Slice2 + 1] = 1

            # 核的维数加2
            Dim = Dim + 2

        # 旋转核以获得八个（该分支为四个）方向
        if Direction == 'N':
            pass
        elif Direction == 'S':
            CenterKernel = np.rot90(CenterKernel, 2)
        elif Direction == 'E':
            CenterKernel = np.rot90(CenterKernel, -1)
        elif Direction == 'W':
            CenterKernel = np.rot90(CenterKernel, 1)

        # 赋值给结果
        Result = CenterKernel

    # 该分支为45度的核
    elif (Direction == 'NE') or (Direction == 'SE') or (Direction == 'SW') or (Direction == 'NW'):

        # 确定中心核（种子）
        CenterKernel = np.array([[2, 0, 0],
                                 [0, -1 * Center, 0],
                                 [0, 0, 0]])

        # 开始生长
        while Dim < Size:

            # 增大中心核
            CenterKernel = np.pad(CenterKernel, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

            # 获得此时中心核的尺寸
            height, width = CenterKernel.shape

            # 分奇偶两种情况赋值，该分支为奇情况
            if (((width - 1) / 2) + 1) % 2 == 0:

                # 求出此时核的周长
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                Slice = int(((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                CenterKernel[0, 0:Slice] = 2
                CenterKernel[0:Slice, 0] = 2

            # 该分支为偶情况
            else:

                # 偶情况要退化成上一次核的的周长
                width = width - 2
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                Slice = int(((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                CenterKernel[0, 0:Slice] = 2
                CenterKernel[0:Slice, 0] = 2

                # 核的这些位置赋值为1
                CenterKernel[0, Slice] = 1
                CenterKernel[Slice, 0] = 1

            # 核的维数加2
            Dim = Dim + 2

        # 旋转核以获得八个（该分支为四个）方向
        if Direction == 'NW':
            pass
        elif Direction == 'SE':
            CenterKernel = np.rot90(CenterKernel, 2)
        elif Direction == 'NE':
            CenterKernel = np.rot90(CenterKernel, -1)
        elif Direction == 'SW':
            CenterKernel = np.rot90(CenterKernel, 1)

        # 赋值给结果
        Result = CenterKernel

    # 返回时做归一化处理
    return Result / Center


def GradientKernelGenerator(Size, Direction):
    """
            用来生成八个方向的探测梯度模的计算核，使用的方法是从中心向四周生长
            的方法。生成的有两种基本核，正方向的（N），斜向45度的（NW），通过
            旋转来达到八个方向。方法与 CutKernelGenerator() 类似，希望可以
            合写成一个基本method，注释不做赘述
            参数描述：
                Size      - 探测梯度模核的大小（尺寸）
                Direction - 探测梯度模核的方向
    """

    Size = int(Size / 2) * 2 + 1
    if Size < 3:
        Size = 3
    Dim = int((Size - 1) / 2)

    if (Direction == 'N') or (Direction == 'S') or (Direction == 'W') or (Direction == 'E'):
        Kernel1 = np.array([[-1]])
        Kernel1 = np.pad(Kernel1, ((Dim, Dim), (Dim, Dim)), 'constant', constant_values=(0, 0))
        height, width = Kernel1.shape
        Slice = int(((width - 1) / 2))
        Kernel1[0, Slice] = 1

        if Direction == 'N':
            pass
        elif Direction == 'S':
            Kernel1 = np.rot90(Kernel1, 2)
        elif Direction == 'E':
            Kernel1 = np.rot90(Kernel1, -1)
        elif Direction == 'W':
            Kernel1 = np.rot90(Kernel1, 1)
        Kernel = Kernel1

    elif (Direction == 'NE') or (Direction == 'SE') or (Direction == 'SW') or (Direction == 'NW'):
        Kernel1 = np.array([[-1]])
        Kernel1 = np.pad(Kernel1, ((Dim, Dim), (Dim, Dim)), 'constant', constant_values=(0, 0))
        Kernel1[0, 0] = 1

        if Direction == 'NW':
            pass
        elif Direction == 'SE':
            Kernel1 = np.rot90(Kernel1, 2)
        elif Direction == 'NE':
            Kernel1 = np.rot90(Kernel1, -1)
        elif Direction == 'SW':
            Kernel1 = np.rot90(Kernel1, 1)
        Kernel = Kernel1

    return Kernel


def GradientDetector(img, Size, Direction, mode='abs'):
    """
            累加梯度探测器，使用 GradientKernelGenerator() 生成的梯度模探测核进行累加梯度计算。
            参数描述：
                img       - 被探测图像
                Size      - 探测梯度模核的大小（尺寸）
                Direction - 探测梯度模核的方向
                mode      - 累加时是否取绝对值再累加
    """

    # 确定梯度模探测核大小，最小不小于3*3
    Size = int(Size / 2) * 2 + 1
    if Size < 3:
        Size = 3

    # 初始化
    img = np.array(img).astype(np.float32)
    Gradient = np.zeros(img.shape)
    Dim = 1

    # 累加梯度探测
    while Dim < Size:
        Dim = Dim + 2
        Kernel = GradientKernelGenerator(Size=Dim, Direction=Direction)
        if mode == 'abs':
            Nabla = abs(np.array(cv2.filter2D(img, -1, kernel=Kernel)))
        elif mode == 'nonabs':
            Nabla = np.array(cv2.filter2D(img, -1, kernel=Kernel))
        Gradient = Gradient + Nabla

    # 累加梯度为绝对值
    return abs(Gradient)


def Normalized(img, Max, Min):
    """
            数据归一化的method。
            参数描述：
                img - 被归一化图像
                Max - 归一化最大值
                Min - 归一化最小值
    """

    # 初始化
    img = np.array(img)

    # 要归一的范围的最大值
    Ymax = Max

    # 要归一的范围的最小值
    Ymin = Min

    # 所有数据中最大的
    Xmax = np.max(img)

    # 所有数据中最小的
    Xmin = np.min(img)

    # 归一化计算
    Normalizedimg = (Ymax - Ymin) * (img - Xmin) / (Xmax - Xmin) + Ymin
    return Normalizedimg


def AsynchronousController(img, Ctrlimg, BreakPoint, BreakPoint2=0, Mode='S&E'):
    """
            异步控制器，控制该点该方向在本次中是否迭代。
            参数描述：
                img         - 被控制的方向梯度模图像
                Ctrlimg     - 控制图像
                BreakPoint  - 控制断点阈值1
                BreakPoint2 - 控制断点阈值2
                Mode        - 小于等于(S&E)，大于等于(B&E)，大于小于(B&S)
    """

    # 执行控制
    if Mode == 'S&E':
        Ctrlimg[Ctrlimg <= BreakPoint] = 0
    elif Mode == 'B&E':
        Ctrlimg[Ctrlimg >= BreakPoint] = 0
    elif Mode == 'B&S':
        Ctrlimg[(Ctrlimg >= BreakPoint) & (Ctrlimg <= BreakPoint2)] = 0
    Ctrlimg[Ctrlimg != 0] = 1

    # 相乘以实现控制
    img = Ctrlimg * img
    return img


def SharpNoiseSuppression(Image, Threshold, KernelSize=3, Size=5):
    Image = np.array(Image).astype(np.float32)
    KernelSize = int(KernelSize / 2) * 2 + 1
    if KernelSize < 3:
        KernelSize = 3
    Center = int(KernelSize / 2) + 1
    Kernel = np.ones((KernelSize, KernelSize))
    Kernel[Center-1, Center-1] = -1
    SharpDetectImage = np.array(Normalized(abs(cv2.filter2D(Image, -1, kernel=Kernel)), Max=1, Min=0))

    Ctrlimg = SharpDetectImage.copy()
    Ctrlimg[Ctrlimg <= Threshold] = 0
    Ctrlimg[Ctrlimg != 0] = 1

    RevCtrlimg = SharpDetectImage.copy()
    RevCtrlimg[RevCtrlimg > Threshold] = 0
    RevCtrlimg[RevCtrlimg != 0] = 1

    for _ in range(5):
        ImageBlur = np.array(cv2.medianBlur(src=Image, ksize=Size))
    Result = (Image * RevCtrlimg) + (ImageBlur * Ctrlimg)
    return Result


def asyanisodiff2D(Image, IterNumber=10, DeltaT=1 / 7, Kappa=0.05, KernelSize=3, GaussSigma=5, NoiseDetectSize=None,
                   NoiseCutoff=0.088, Option=2, EdgeKeeping='yes', EdgeDetectSize=None, EdgeCutoffLow=0.35,
                   EdgeCutoffHigh=0.8, Dimension=1):
    """
    ASYANISODIFF2D Catte_PM模型的各向异性异步扩散
    该函数执行灰度图像上的各向异性异步扩散，被认为是一个二维的网络结构的 （KernelSize^2）-1 个相邻节点的扩散传导
    参数描述：
        Image           - 原始图像
        IterNumber      - 迭代次数
        DeltaT          - 积分常数（0 <=  delta_t <= 1/7),通常情况下，由于数值稳定性，此参数设置为它的最大值
        Kappa           - 控制传导的梯度模阈值，控制平滑,越大越平滑
        KernelSize      - 卷积核的大小，一般为 3*3 (3)
        GaussSigma      - 高斯平滑的高斯核方差大小
        NoiseDetectSize - 噪声探测的范围，通常设置为卷积核大小的两倍
        NoiseCutoff     - 噪声切除的阈值，其值在 0~1 之间，为噪声探测值经过归一化之后的切除阈值，不建议超过 0.2
        Option          - 传导系数函数选择（Perona & Malik 提出）：
                           1 - c(x,y,t) = exp(-(nablaI/Kappa)**2)
                           2 - c(x,y,t) = 1/(1 + (nablaI/Kappa)**2)
    """

    # 初始化
    if NoiseDetectSize is None:
        NoiseDetectSize = KernelSize * 2
    if EdgeDetectSize is None:
        EdgeDetectSize = NoiseDetectSize

    Image = np.array(Image).astype(np.float32)

    # 中心像素距离
    dx = 1
    dy = 1
    dd = np.sqrt(2)

    # 二维卷积掩模-8个方向上的梯度差分
    hN = CutKernelGenerator(Size=KernelSize, Direction='N')
    hS = CutKernelGenerator(Size=KernelSize, Direction='S')
    hW = CutKernelGenerator(Size=KernelSize, Direction='W')
    hE = CutKernelGenerator(Size=KernelSize, Direction='E')
    hNE = CutKernelGenerator(Size=KernelSize, Direction='NE')
    hSE = CutKernelGenerator(Size=KernelSize, Direction='SE')
    hSW = CutKernelGenerator(Size=KernelSize, Direction='SW')
    hNW = CutKernelGenerator(Size=KernelSize, Direction='NW')

    # 各向异性扩散
    DiffImage = Image
    for t in range(IterNumber):

        Gaussimg = cv2.filter2D(DiffImage, -1, kernel=GaussianKernel2D(Sigma=GaussSigma, Size=4 * GaussSigma))
        Gaussimg = np.array(Gaussimg).astype(np.float32)
        MedianGaussimg = cv2.medianBlur(src=Gaussimg, ksize=7)

        NoiseGradN = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='N'), Max=1, Min=0)
        NoiseGradS = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='S'), Max=1, Min=0)
        NoiseGradW = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='W'), Max=1, Min=0)
        NoiseGradE = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='E'), Max=1, Min=0)
        NoiseGradNE = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='NE'), Max=1, Min=0)
        NoiseGradSE = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='SE'), Max=1, Min=0)
        NoiseGradSW = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='SW'), Max=1, Min=0)
        NoiseGradNW = Normalized(GradientDetector(Gaussimg, Size=NoiseDetectSize, Direction='NW'), Max=1, Min=0)

        if EdgeKeeping == 'yes':
            EdgeGradN = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='N'), Max=1, Min=0)
            EdgeGradS = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='S'), Max=1, Min=0)
            EdgeGradW = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='W'), Max=1, Min=0)
            EdgeGradE = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='E'), Max=1, Min=0)
            EdgeGradNE = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='NE'), Max=1, Min=0)
            EdgeGradSE = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='SE'), Max=1, Min=0)
            EdgeGradSW = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='SW'), Max=1, Min=0)
            EdgeGradNW = Normalized(GradientDetector(MedianGaussimg, Size=EdgeDetectSize, Direction='NW'), Max=1, Min=0)

        nablaN = cv2.filter2D(DiffImage, -1, kernel=hN)  # -1:目标图像与原图像深度保持一致
        nablaS = cv2.filter2D(DiffImage, -1, kernel=hS)
        nablaW = cv2.filter2D(DiffImage, -1, kernel=hW)
        nablaE = cv2.filter2D(DiffImage, -1, kernel=hE)
        nablaNE = cv2.filter2D(DiffImage, -1, kernel=hNE)
        nablaSE = cv2.filter2D(DiffImage, -1, kernel=hSE)
        nablaSW = cv2.filter2D(DiffImage, -1, kernel=hSW)
        nablaNW = cv2.filter2D(DiffImage, -1, kernel=hNW)

        GaussnablaN = cv2.filter2D(Gaussimg, -1, kernel=hN)  # -1:目标图像与原图像深度保持一致
        GaussnablaS = cv2.filter2D(Gaussimg, -1, kernel=hS)
        GaussnablaW = cv2.filter2D(Gaussimg, -1, kernel=hW)
        GaussnablaE = cv2.filter2D(Gaussimg, -1, kernel=hE)
        GaussnablaNE = cv2.filter2D(Gaussimg, -1, kernel=hNE)
        GaussnablaSE = cv2.filter2D(Gaussimg, -1, kernel=hSE)
        GaussnablaSW = cv2.filter2D(Gaussimg, -1, kernel=hSW)
        GaussnablaNW = cv2.filter2D(Gaussimg, -1, kernel=hNW)

        # 扩散函数
        if Option == 1:
            cN = np.exp(-(GaussnablaN / Kappa) ** 2)
            cS = np.exp(-(GaussnablaS / Kappa) ** 2)
            cW = np.exp(-(GaussnablaW / Kappa) ** 2)
            cE = np.exp(-(GaussnablaE / Kappa) ** 2)
            cNE = np.exp(-(GaussnablaNE / Kappa) ** 2)
            cSE = np.exp(-(GaussnablaSE / Kappa) ** 2)
            cSW = np.exp(-(GaussnablaSW / Kappa) ** 2)
            cNW = np.exp(-(GaussnablaNW / Kappa) ** 2)
        elif Option == 2:
            cN = 1. / (1 + (GaussnablaN / Kappa) ** 2)
            cS = 1. / (1 + (GaussnablaS / Kappa) ** 2)
            cW = 1. / (1 + (GaussnablaW / Kappa) ** 2)
            cE = 1. / (1 + (GaussnablaE / Kappa) ** 2)
            cNE = 1. / (1 + (GaussnablaNE / Kappa) ** 2)
            cSE = 1. / (1 + (GaussnablaSE / Kappa) ** 2)
            cSW = 1. / (1 + (GaussnablaSW / Kappa) ** 2)
            cNW = 1. / (1 + (GaussnablaNW / Kappa) ** 2)

        nablaN = AsynchronousController(img=nablaN, Ctrlimg=NoiseGradN, BreakPoint=NoiseCutoff)
        nablaS = AsynchronousController(img=nablaS, Ctrlimg=NoiseGradS, BreakPoint=NoiseCutoff)
        nablaW = AsynchronousController(img=nablaW, Ctrlimg=NoiseGradW, BreakPoint=NoiseCutoff)
        nablaE = AsynchronousController(img=nablaE, Ctrlimg=NoiseGradE, BreakPoint=NoiseCutoff)
        nablaNE = AsynchronousController(img=nablaNE, Ctrlimg=NoiseGradNE, BreakPoint=NoiseCutoff)
        nablaSE = AsynchronousController(img=nablaSE, Ctrlimg=NoiseGradSE, BreakPoint=NoiseCutoff)
        nablaSW = AsynchronousController(img=nablaSW, Ctrlimg=NoiseGradSW, BreakPoint=NoiseCutoff)
        nablaNW = AsynchronousController(img=nablaNW, Ctrlimg=NoiseGradNW, BreakPoint=NoiseCutoff)

        if EdgeKeeping == 'yes':
            nablaN = AsynchronousController(img=nablaN, Ctrlimg=EdgeGradN, BreakPoint=EdgeCutoffLow,
                                            BreakPoint2=EdgeCutoffHigh, Mode='B&S')
            nablaS = AsynchronousController(img=nablaS, Ctrlimg=EdgeGradS, BreakPoint=EdgeCutoffLow,
                                            BreakPoint2=EdgeCutoffHigh, Mode='B&S')
            nablaW = AsynchronousController(img=nablaW, Ctrlimg=EdgeGradW, BreakPoint=EdgeCutoffLow,
                                            BreakPoint2=EdgeCutoffHigh, Mode='B&S')
            nablaE = AsynchronousController(img=nablaE, Ctrlimg=EdgeGradE, BreakPoint=EdgeCutoffLow,
                                            BreakPoint2=EdgeCutoffHigh, Mode='B&S')
            nablaNE = AsynchronousController(img=nablaNE, Ctrlimg=EdgeGradNE, BreakPoint=EdgeCutoffLow,
                                             BreakPoint2=EdgeCutoffHigh, Mode='B&S')
            nablaSE = AsynchronousController(img=nablaSE, Ctrlimg=EdgeGradSE, BreakPoint=EdgeCutoffLow,
                                             BreakPoint2=EdgeCutoffHigh, Mode='B&S')
            nablaSW = AsynchronousController(img=nablaSW, Ctrlimg=EdgeGradSW, BreakPoint=EdgeCutoffLow,
                                             BreakPoint2=EdgeCutoffHigh, Mode='B&S')
            nablaNW = AsynchronousController(img=nablaNW, Ctrlimg=EdgeGradNW, BreakPoint=EdgeCutoffLow,
                                             BreakPoint2=EdgeCutoffHigh, Mode='B&S')

        # 离散偏微分方程的解决方案
        Delta = DeltaT * (1 / (dy ** 2) * cN * nablaN +
                          1 / (dy ** 2) * cS * nablaS +
                          1 / (dx ** 2) * cW * nablaW +
                          1 / (dx ** 2) * cE * nablaE +
                          1 / (dd ** 2) * cNE * nablaNE +
                          1 / (dd ** 2) * cSE * nablaSE +
                          1 / (dd ** 2) * cSW * nablaSW +
                          1 / (dd ** 2) * cNW * nablaNW)

        DiffImage = DiffImage + Delta

    return DiffImage
