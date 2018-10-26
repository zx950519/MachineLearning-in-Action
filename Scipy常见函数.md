# Scipy常用函数 

# Scipy
    Scipy包包含许多专注于科学计算中的常见问题的工具箱。
    它的子模块对应于不同的应用比如插值、积分、优化、图像处理、统计和特殊功能等。
    Scipy可以与其他标准科学计算包相对比，比如GSL (C和C++的GNU科学计算包),或者Matlab的工具箱。
    scipy是Python中科学程序的核心程序包；这意味着有效的操作numpy数组，
    因此，Numpy和Scipy可以一起工作。

# 参考源
    https://wizardforcel.gitbooks.io/scipy-lecture-notes/content/4.html
<br>

## I/O
    略

## 特殊函数
    例如:
        贝塞尔函数:比如scipy.special.jn() (第n个整型顺序的贝塞尔函数)
        椭圆函数:(scipy.special.ellipj() Jacobian椭圆函数, ...)
        Gamma 函数: scipy.special.gamma(), 也要注意 scipy.special.gammaln() 将给出更高准确数值的 Gamma的log。
        Erf, 高斯曲线的面积：scipy.special.erf()
    官方参考:https://docs.scipy.org/doc/scipy/reference/special.html#scipy.special\

## 线性代数
    scipy.linalg.=numpy.linalg.
    略
    
## 快速傅里叶变换
    scipy.fftpack
    官方参考:https://docs.scipy.org/doc/scipy/reference/fftpack.html#scipy.fftpack
    
## 优化及拟合
    优化是寻找最小化或等式的数值解的问题,
    该模块提供了函数最小化（标量或多维度）、曲线拟合和求根的有用算法。
        
        def f(x):
            return x**2 + 10*np.sin(x)
            
        from scipy import optimize
        x = np.arange(-10, 10, 0.05)
        plt.plot(x, f(x))
        plt.show()
    
    寻找标量函数的最小值:
        #使用梯度下降方法，但是可能因为初始点的选择落入局部最优
        print(optimize.fmin_bfgs(f, 3, disp=0))
        #暴力求解每个点，以找到全局最优，但是时间开销大
        grid = (-10, 10, 0.1)
        print(optimize.brute(f, (grid,)))
        #使用模拟退火求解，并设定求解定义域
        print(optimize.fminbound(f, -2, 10))
    
    寻找标量函数的根:
        print(optimoze.fsolve(f, -2.5)
        
    曲线拟合:
        def f2(x, a, b):
            return a*x**2 + b*np.sin(x)
            
        xdata = np.linspace(-10, 10, num=20)
        ydata = f(xdata) + np.random.randn(xdata.size)
        guess = [2, 2]
        params, params_covariance = optimize.curve_fit(f2, xdata, ydata, guess)
        print(params)
        print(params_covariance)
        
## 插值
        插值-在离散数据的基础上补插连续函数，使得这条连续曲线通过全部给定的离散数据点
        #生成数据点、噪音
        measured_time = np.linspace(0, 1, 10)
        noise = (np.random.random(10)*2 - 1) * 1e-1
        measures = np.sin(2*np.pi*measured_time) + noise
        
        from scipy.interpolate import interp1d
        linear_interp = interp1d(measured_time, measures)
        #线性插值
        computed_time = np.linspace(0, 1, 50)
        linear_results = linear_interp(computed_time)
        #立方插值
        cubic_interp = interp1d(measured_time, measures, kind='cubic')
        cubic_results = cubic_interp(computed_time)
        #绘图
        plt.plot(np.linspace(0, 1, 50), linear_results)
        plt.plot(np.linspace(0, 1, 50), cubic_results)
        plt.show()

## 数值积分
        常见积分:
        from scipy.integrate import quad
        res, err = quad(np.sin, 0, np.pi/2)
        print(res)
