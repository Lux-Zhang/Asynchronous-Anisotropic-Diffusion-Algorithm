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

        Used to generate a two-dimensional Gaussian kernel, because cv2.GaussianBlur()
        is not a simple Gaussian filter. The specific method is to first determine the
        size of the Gaussian kernel and use cv2.getGaussianKernel() to generate two
        identical Gaussian kernels. Then kx times the transpose of ky.
        Parameter Description:
            Size - the size (dimensions) of the Gaussian kernel
            Sigma - Variance of Gaussian kernel
    """

    # 确定高斯核大小，最小不小于5*5
    # Determine the size of the Gaussian kernel, the minimum is not less than 5*5
    Size = int(Size / 2) * 2 + 1
    if Size < 5:
        Size = 5

    # 生成两个一样的一维高斯核，若需要x方向和y方向方差不一样，可生成两个不一样的高斯核
    # Generate two identical one-dimensional Gaussian kernels. If you need to
    # have different variances in the x and y directions, you can generate two
    # different Gaussian kernels
    KernelX = cv2.getGaussianKernel(Size, Sigma)
    KernelY = cv2.getGaussianKernel(Size, Sigma)

    # kx乘ky的转置
    # transpose of kx times ky
    Result = np.multiply(KernelX, np.transpose(KernelY))
    return Result


def CutKernelGenerator(Size, Direction):
    """
        用来生成八个方向的各向异性梯度模的计算核，使用的方法是从中心向四周生长的方法。
        生成的有两种基本核，正方向的（N），斜向45度的（NW），通过旋转来达到八个方向。
        参数描述：
            Size      - 梯度模核的大小（尺寸）
            Direction - 梯度模核的方向

        The computational kernel used to generate anisotropic gradient modes
        in eight directions, using the method of growing from the center to
        the periphery. There are two basic kernels generated, the positive
        direction (N), the oblique 45 degree (NW), and the eight directions
        are achieved by rotation.
        Parameter Description:
            Size - the size (dimensions) of the gradient kernel
            Direction - Direction of the gradient kernel
    """

    # 确定梯度模核大小，最小不小于3*3
    # Determine the size of the gradient kernel, the minimum is not less than 3*3
    Size = int(Size / 2) * 2 + 1
    if Size < 3:
        Size = 3
    Dim = 3

    # 确定核的中心值
    # Determine the center value of the kernel
    Center = ((Size ** 2) - 1) * 2 / 8

    # 以下分为两个大分支，正向的核和45度的核，该分支为正向的核
    # The following is divided into two large branches,
    # the forward kernel and the 45-degree kernel, this
    # branch is the forward kernel
    if (Direction == 'N') or (Direction == 'S') or (Direction == 'W') or (Direction == 'E'):

        # 确定中心核（种子）
        # Determine the central core (seed)
        CenterKernel = np.array([[0, 2, 0],
                                 [0, -1 * Center, 0],
                                 [0, 0, 0]])

        # 开始生长
        # start growing
        while Dim < Size:

            # 增大中心核
            # Increase the center core
            CenterKernel = np.pad(CenterKernel, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

            # 获得此时中心核的尺寸
            # Get the size of the center core at this time
            height, width = CenterKernel.shape

            # 分奇偶两种情况赋值，该分支为奇情况
            # Assign the assignment in two cases of parity and even, the branch is odd
            if (((width - 1) / 2) + 1) % 2 == 0:

                # 求出此时核的周长
                # Find the perimeter of the core at this time
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                # Find the factor based on the perimeter
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                # Find the two slice positions based on the factor
                Slice1 = int(((width - 1) / 2) - ((Factor - 1) / 2))
                Slice2 = int(((width - 1) / 2) + ((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                # These positions of the core are assigned a value of 2
                CenterKernel[0, Slice1:Slice2] = 2

            # 该分支为偶情况
            # This branch is an even case
            else:

                # 偶情况要退化成上一次核的的周长
                # The even case degenerates to the perimeter of the last kernel
                width = width - 2
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                # Find the factor based on the perimeter
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                # Find the two slice positions based on the factor
                Slice1 = int(((width - 1) / 2) - ((Factor - 1) / 2))
                Slice2 = int(((width - 1) / 2) + ((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                # These positions of the core are assigned a value of 2
                CenterKernel[0, Slice1 + 1:Slice2 + 1] = 2

                # 核的这些位置赋值为1
                # These positions of the core are assigned a value of 1
                CenterKernel[0, Slice1] = 1
                CenterKernel[0, Slice2 + 1] = 1

            # 核的尺寸加2
            # kernel size plus 2
            Dim = Dim + 2

        # 旋转核以获得八个（该分支为四个）方向
        # Rotate the kernel to get eight (four for this branch) orientations
        if Direction == 'N':
            pass
        elif Direction == 'S':
            CenterKernel = np.rot90(CenterKernel, 2)
        elif Direction == 'E':
            CenterKernel = np.rot90(CenterKernel, -1)
        elif Direction == 'W':
            CenterKernel = np.rot90(CenterKernel, 1)

        # 赋值给结果
        # assign to the result
        Result = CenterKernel

    # 该分支为45度的核
    # This branch is the 45 degree kernel
    elif (Direction == 'NE') or (Direction == 'SE') or (Direction == 'SW') or (Direction == 'NW'):

        # 确定中心核（种子）
        # Determine the central core (seed)
        CenterKernel = np.array([[2, 0, 0],
                                 [0, -1 * Center, 0],
                                 [0, 0, 0]])

        # 开始生长
        # start growing
        while Dim < Size:

            # 增大中心核
            # Increase the center core
            CenterKernel = np.pad(CenterKernel, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

            # 获得此时中心核的尺寸
            # Get the size of the center core at this time
            height, width = CenterKernel.shape

            # 分奇偶两种情况赋值，该分支为奇情况
            # Assign the assignment in two cases of parity and even, the branch is odd
            if (((width - 1) / 2) + 1) % 2 == 0:

                # 求出此时核的周长
                # Find the perimeter of the core at this time
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                # Find the factor based on the perimeter
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                # Find the two slice positions based on the factor
                Slice = int(((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                # Find the two slice positions based on the factor
                CenterKernel[0, 0:Slice] = 2
                CenterKernel[0:Slice, 0] = 2

            # 该分支为偶情况
            # This branch is an even case
            else:

                # 偶情况要退化成上一次核的的周长
                # The even case degenerates to the perimeter of the last kernel
                width = width - 2
                Perimeter = 4 * (width - 1)

                # 根据周长求出因子
                # Find the factor based on the perimeter
                Factor = Perimeter / 8

                # 根据因子求出两个切片位置
                # Find the two slice positions based on the factor
                Slice = int(((Factor - 1) / 2) + 1)

                # 核的这些位置赋值为2
                # These positions of the core are assigned a value of 2
                CenterKernel[0, 0:Slice] = 2
                CenterKernel[0:Slice, 0] = 2

                # 核的这些位置赋值为1
                # These positions of the core are assigned a value of 1
                CenterKernel[0, Slice] = 1
                CenterKernel[Slice, 0] = 1

            # 核的尺寸加2
            # kernel size plus 2
            Dim = Dim + 2

        # 旋转核以获得八个（该分支为四个）方向
        # Rotate the kernel to get eight (four for this branch) orientations
        if Direction == 'NW':
            pass
        elif Direction == 'SE':
            CenterKernel = np.rot90(CenterKernel, 2)
        elif Direction == 'NE':
            CenterKernel = np.rot90(CenterKernel, -1)
        elif Direction == 'SW':
            CenterKernel = np.rot90(CenterKernel, 1)

        # 赋值给结果
        # assign to the result
        Result = CenterKernel

    # 返回时做归一化处理
    # Normalize when returning
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

            The computational kernel used to generate probe gradient
            modes in eight directions by growing from the center to the
            periphery methods. There are two basic kernels generated,
            positive direction (N), oblique 45 degrees (NW), through
            rotate to reach eight directions. The method is similar to
            CutKernelGenerator(), hopefully refactor into a basic method,
            the comments will not go into details
            Parameter Description:
                Size - the size (dimension) of the probe gradient kernel
                Direction - Direction of the probe gradient kernel
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

            Accumulated gradient detector, which uses the gradient modulo detection kernel
            generated by GradientKernelGenerator() for cumulative gradient calculation.
            Parameter Description:
                img - the detected image
                Size - the size (dimension) of the probe gradient kernel
                Direction - Direction of the probe gradient kernel
                mode - whether to take the absolute value and then accumulate when accumulating
    """

    # 确定梯度模探测核大小，最小不小于3*3
    # Determine the size of the gradient
    # mode detection kernel, the minimum
    # is not less than 3*3
    Size = int(Size / 2) * 2 + 1
    if Size < 3:
        Size = 3

    # 初始化
    # initialize
    img = np.array(img).astype(np.float32)
    Gradient = np.zeros(img.shape)
    Dim = 1

    # 累加梯度探测
    # Accumulate gradient detection
    while Dim < Size:
        Dim = Dim + 2
        Kernel = GradientKernelGenerator(Size=Dim, Direction=Direction)
        if mode == 'abs':
            Nabla = abs(np.array(cv2.filter2D(img, -1, kernel=Kernel)))
        elif mode == 'nonabs':
            Nabla = np.array(cv2.filter2D(img, -1, kernel=Kernel))
        Gradient = Gradient + Nabla

    # 累加梯度为绝对值
    # Accumulate gradient as absolute value
    return abs(Gradient)


def Normalized(img, Max, Min):
    """
            数据归一化的method。
            参数描述：
                img - 被归一化图像
                Max - 归一化最大值
                Min - 归一化最小值

            Data normalization method.
            Parameter Description:
                img - the normalized image
                Max - Normalized maximum value
                Min - Normalized minimum value
    """

    # 初始化
    # initialize
    img = np.array(img)

    # 要归一的范围的最大值
    # The maximum value of the range to normalize
    Ymax = Max

    # 要归一的范围的最小值
    # Minimum value of the range to normalize
    Ymin = Min

    # 所有数据中最大的
    # The largest of all data
    Xmax = np.max(img)

    # 所有数据中最小的
    # smallest of all data
    Xmin = np.min(img)

    # 归一化计算
    # Normalize calculation
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

            An asynchronous controller that controls
            whether the direction at this point is
            iterated this time.
            Parameter Description:
                img - the controlled directional gradient norm image
                Ctrlimg - Control image
                BreakPoint - Controls the breakpoint threshold of 1
                BreakPoint2 - Controls breakpoint threshold 2
                Mode - less than or equal to (S&E), greater than or
                       equal to (B&E), greater than or less than (B&S)
    """

    # 执行控制
    # execute control
    if Mode == 'S&E':
        Ctrlimg[Ctrlimg <= BreakPoint] = 0
    elif Mode == 'B&E':
        Ctrlimg[Ctrlimg >= BreakPoint] = 0
    elif Mode == 'B&S':
        Ctrlimg[(Ctrlimg >= BreakPoint) & (Ctrlimg <= BreakPoint2)] = 0
    Ctrlimg[Ctrlimg != 0] = 1

    # 相乘以实现控制
    # Multiply for control
    img = Ctrlimg * img
    return img


def asyanisodiff2D(Image, IterNumber=10, DeltaT=1 / 7, Kappa=0.05, KernelSize=3, GaussSigma=5, NoiseDetectSize=None,
                   NoiseCutoff=0.088, Option=2, EdgeKeeping='yes', EdgeDetectSize=None, EdgeCutoffLow=0.35,
                   EdgeCutoffHigh=0.8):
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

        Anisotropic Asynchronous Diffusion for ASYANISODIFF2D Catte_PM Model
        This function performs anisotropic asynchronous diffusion on grayscale images, considered as a
        two-dimensional network structure of (KernelSize^2)-1 adjacent nodes of diffusion conduction
        Parameter Description:
            Image - the original image
            IterNumber - the number of iterations
            DeltaT - integral constant (0 <= delta_t <= 1/7), normally, due to numerical stability,
                     this parameter is set to its maximum value
            Kappa - Gradient modulo threshold that controls conduction, controls smoothing, the larger the smoother
            KernelSize - the size of the convolution kernel, typically 3*3 (3)
            GaussSigma - Gaussian kernel variance magnitude for Gaussian smoothing
            NoiseDetectSize - the range of noise detection, usually set to twice the size of the convolution kernel
            NoiseCutoff - the threshold of noise cutoff, its value is between 0 and 1, it is the cutoff
                          threshold after the noise detection value is normalized, it is not recommended to exceed 0.2
            Option - Conductivity function selection (by Perona & Malik):
                           1 - c(x,y,t) = exp(-(nablaI/Kappa)**2)
                           2 - c(x,y,t) = 1/(1 + (nablaI/Kappa)**2)
    """

    # 初始化
    # initialize
    if NoiseDetectSize is None:
        NoiseDetectSize = KernelSize * 2
    if EdgeDetectSize is None:
        EdgeDetectSize = NoiseDetectSize

    Image = np.array(Image).astype(np.float32)

    # 中心像素距离
    # center pixel distance
    dx = 1
    dy = 1
    dd = np.sqrt(2)

    # 二维卷积掩模-8个方向上的梯度差分
    # 2D convolution mask - gradient difference in 8 directions
    hN = CutKernelGenerator(Size=KernelSize, Direction='N')
    hS = CutKernelGenerator(Size=KernelSize, Direction='S')
    hW = CutKernelGenerator(Size=KernelSize, Direction='W')
    hE = CutKernelGenerator(Size=KernelSize, Direction='E')
    hNE = CutKernelGenerator(Size=KernelSize, Direction='NE')
    hSE = CutKernelGenerator(Size=KernelSize, Direction='SE')
    hSW = CutKernelGenerator(Size=KernelSize, Direction='SW')
    hNW = CutKernelGenerator(Size=KernelSize, Direction='NW')

    # 各向异性扩散
    # anisotropic diffusion
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
        nablaS = cv2.filter2D(DiffImage, -1, kernel=hS)  # -1: The target image has the same depth as the original image
        nablaW = cv2.filter2D(DiffImage, -1, kernel=hW)
        nablaE = cv2.filter2D(DiffImage, -1, kernel=hE)
        nablaNE = cv2.filter2D(DiffImage, -1, kernel=hNE)
        nablaSE = cv2.filter2D(DiffImage, -1, kernel=hSE)
        nablaSW = cv2.filter2D(DiffImage, -1, kernel=hSW)
        nablaNW = cv2.filter2D(DiffImage, -1, kernel=hNW)

        GaussnablaN = cv2.filter2D(Gaussimg, -1, kernel=hN)
        GaussnablaS = cv2.filter2D(Gaussimg, -1, kernel=hS)
        GaussnablaW = cv2.filter2D(Gaussimg, -1, kernel=hW)
        GaussnablaE = cv2.filter2D(Gaussimg, -1, kernel=hE)
        GaussnablaNE = cv2.filter2D(Gaussimg, -1, kernel=hNE)
        GaussnablaSE = cv2.filter2D(Gaussimg, -1, kernel=hSE)
        GaussnablaSW = cv2.filter2D(Gaussimg, -1, kernel=hSW)
        GaussnablaNW = cv2.filter2D(Gaussimg, -1, kernel=hNW)

        # 扩散函数
        # spread function
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
        # Solution of discrete partial differential equations
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
