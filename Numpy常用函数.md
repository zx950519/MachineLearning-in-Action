# Numpy常用函数 

# 参考源
    https://blog.csdn.net/u013457382/article/details/50828646
    http://www.numpy.org/
    https://blog.csdn.net/pipisorry/article/details/39235753
## 高级Numpy请参考
    https://wizardforcel.gitbooks.io/scipy-lecture-notes/content/7.html
<br>

## numpy.生成
    numpy.linespace(start=起始点;stop=终止点;num=数量):生成等差数列
    numpy.array([1,2,3]或[[1,2],[3,4]])：创建数组
    numpy.zeros(行数,列数)：创建一个全0的数组
    numpy.ones(行数,列数:创建一个全1的数组
    numpy.empty(行数,列数):创建一个随机的数组
    -numpy.arange(起始点(包含),终止点(不包含),步长)：返回等差数列
    -numpy.eye(行数,生成在哪条对角线上)：生成对角矩阵

## numpy.随机
    -numpy.random.rand(无参生成单个随机数,有参则生成指定格式的随机数序列,例如5或2,3)：返回随机数(均值分布)序列
    -numpy.random.randn(无参生成单个随机数,有参则生成指定格式的随机数序列,例如5或2,3)：返回随机数(正态分布)序列
    -numpy.random.standard_normal(无参生成单个随机数,有参则生成指定格式的随机数序列,例如5或(2,3))：返回随机数(正态分布)序列
    -numpy.random.randint(low=下限,high=上限(不包含),size=5或(2,3))：返回类型指定为int的随机数序列
    -numpy.random.random_integers(low=下限,high=上限(不包含),size=5或(2,3))：返回类型指定为int的随机数序列,该函数与randint()区别在于当high为None时,该函数取[1,low],上面的函数取[0,low]
    -numpy.random.random_sample(size5或(2,3))：生成一个[0,1)之间随机浮点数或N维浮点数组
    -numpy.random.shuffle(a)：对指定数组进行重排序(随机),对于多维数组,只沿着第一条轴打乱顺序
	
## numpy.索引&统计
    -numpy.argmax(a,axis=0/1)：获取最大值的索引
    -numpy.argmin(a,axis=0/1)：获取最小值的索引
    -numpy.argsort(a.axis=0/1)：获取数组值从小到大的索引值
    -numpy.bincount(a)：统计数组元素出现的次数
    -numpy.average(a,axie=0/1,weights=[权重1,权重2,...,权重n])：求加权平均值
    -numpy.var(a)：求方差
    -numpy.sta(a，axis=0/1):求标准差
    -numpy.cov(a,bais=True/False)：求协方差(True表示求均值时除以n),返回协方差矩阵,results[i][j]表示第i个随机变量与第j个随机变量的协方差
    -numpy.corrcoef(a)：求相关系数,返回相关系数矩阵,results[i][j]表示第i个随机变量与第j个随机变量的相关系数
    -numpy.cunprod(a)：数组累乘
    -numpy.nonzero(a)：返回数组或矩阵中不为0的元素下标
    -numpy.mean(a,axis=0/1)：求均值,返回实数;1*n矩阵;m*1矩阵
    -numpy.median(a,axis=0/1)：求中位数,返回实数;1*n矩阵;m*1矩阵

## numpy.三角函数
	-sin/cos/tan/arcsin/arccos/arctan(np.pi/2)
	-hypot(x1,x2)：三角形求斜边
	-degrees(x)：弧度转为度
	-rad2deg(x)：弧度转为度
	-raduabs(x)：度转为弧度
	-deg2rad(x)：度转为弧度

## numpy.双曲函数
	-sinh/cosh/tanh(x)：双曲正弦/双曲余弦/双曲正切
	-arcsinh/arcosh/arctanh(x)：反双曲正弦/反双曲余弦/反双曲正切

## numpy.数值修约
	-numpy.prod(a, axis, dtype, keepdims)：返回指定轴上的数组元素的乘积
	-numpy.sum(a, axis, dtype, keepdims)：返回指定轴上的数组元素的总和
	-numpy.nanprod(a, axis, dtype, keepdims)：返回指定轴上的数组元素的乘积, 将 NaN 视作 1
	-numpy.nansum(a, axis, dtype, keepdims)：返回指定轴上的数组元素的总和, 将 NaN 视作 0
	-numpy.cumprod(a, axis, dtype)：返回沿给定轴的元素的累积乘积
	-numpy.cumsum(a, axis, dtype)：返回沿给定轴的元素的累积总和
	-numpy.nancumprod(a, axis, dtype)：返回沿给定轴的元素的累积乘积, 将 NaN 视作 1
	-numpy.nancumsum(a, axis, dtype)：返回沿给定轴的元素的累积总和, 将 NaN 视作 0
	-numpy.diff(a, n, axis)：计算沿指定轴的第 n 个离散差分
	-numpy.ediff1d(ary, to_end, to_begin)：数组的连续元素之间的差异
	-numpy.gradient(f)：返回 N 维数组的梯度
	-numpy.cross(a, b, axisa, axisb, axisc, axis)：返回两个(数组）向量的叉积
	-numpy.trapz(y, x, dx, axis)：使用复合梯形规则沿给定轴积分
	-numpy.ceil(a)：向上取整
	-numpy.floor(a):向下取整
	-numpy.clip(a,min,max)：根据参数将不在规定范围内的数据规整化

## numpy.指数和对数
	-numpy.exp(x)：计算输入数组中所有元素的指数
	-numpy.expm1(x)：对数组中的所有元素计算 exp(x） - 1
	-numpy.exp2(x)：对于输入数组中的所有 p, 计算 2 ** p
	-numpy.log(x)：计算自然对数
	-numpy.log10(x)：计算常用对数
	-numpy.log2(x)：计算二进制对数
	-numpy.log1p(x)：log(1 + x)
	-numpy.logaddexp(x1, x2)：log2(2**x1 + 2**x2)
	-numpy.logaddexp2(x1, x2)：log(exp(x1) + exp(x2))
	
## numpy.算数运算
	-numpy.add(x1, x2)：对应元素相加
	-numpy.reciprocal(x)：求倒数 1/x
	-numpy.negative(x)：求对应负数
	-numpy.multiply(x1, x2)：求解乘法
	-numpy.divide(x1, x2)：相除 x1/x2
	-numpy.power(x1, x2)：类似于 x1^x2
	-numpy.subtract(x1, x2)：减法
	-numpy.fmod(x1, x2)：返回除法的元素余项
	-numpy.mod(x1, x2)：返回余项
	-numpy.modf(x1)：返回数组的小数和整数部分
	-numpy.remainder(x1, x2)：返回除法余数
	
## numpy.矩阵和向量积
	-numpy.dot(a,b)：求解两个数组的点积
	-numpy.vdot(a,b)：求解两个向量的点积
	-numpy.inner(a,b)：求解两个数组的内积
	-numpy.outer(a,b)：求解两个向量的外积
	-numpy.matmul(a,b)：求解两个数组的矩阵乘积
	-numpy.tensordot(a,b)：求解张量点积
	-numpy.kron(a,b)：计算 Kronecker 乘积
	-numpy.ndim(a)：获取数组/矩阵的秩
	-numpy.shape(a):获取数组/矩阵的维度
	-numpy.size(a)：获取数组的元素总数
    -numpy.transpose(a)：低维数组(二维及其以下)求转置矩阵
    -numpy.ravel(a):二维数组转一维数组(平坦化)
    -numpy.vstack(a,b):按行拼接,也就是竖直方向拼接
	-numpy.hstack(a,b):按列拼接,也就是水平方向拼接
	-numpy.hsplit(a,3):按列分割,也就是横方向分割,参数a为要分割的矩阵,参数3为分成三份
	-numpy.vsplit(a,3):按行分割,也就是纵方向分割,参数a为要分割的矩阵,参数3为分成三份
	-numpy.svd(a):奇异值分解,返回U,sigma,VT
	
## numpy.排序
    -numpy.sort(a,axis=0/1):对数组或矩阵排序(默认升序,实现降序只需要把a改为-a即可)
	-numpy.argsort(a,axis=0/1,kind="quicksort")：返回数组的排序下标
	
## numpy.其他
	-numpy.angle(z, deg)：返回复参数的角度
	-numpy.real(val)：返回数组元素的实部
	-numpy.imag(val)：返回数组元素的虚部
	-numpy.conj(x)：按元素方式返回共轭复数
	-numpy.convolve(a, v, mode)：返回线性卷积
	-numpy.sqrt(x)：平方根
	-numpy.cbrt(x)：立方根
	-numpy.square(x)：平方
	-numpy.absolute(x)：绝对值, 可求解复数
	-numpy.fabs(x)：绝对值
	-numpy.sign(x)：符号函数
	-numpy.maximum(x1, x2)：最大值
	-numpy.minimum(x1, x2)：最小值
	-numpy.nan_to_num(x)：用 0 替换 NaN
	-numpy.interp(x, xp, fp, left, right, period)：线性插值
	-numpy.min(a,axis=0/1):求最小值
    -numpy.max(a,axis=0/1):求最大值

	
## numpy.代数运算
	-numpy.linalg.cholesky(a)：Cholesky分解
	-numpy.linalg.qr(a ,mode)：计算矩阵的QR因式分解
	-numpy.linalg.svd(a ,full_matrices,compute_uv)：奇异值分解
	-numpy.linalg.eig(a)：计算正方形矩阵的特征值和特征向量
	-numpy.linalg.eigh(a, UPLO)：返回 Hermitian 或对称矩阵的特征值和特征向量
	-numpy.linalg.eigvals(a)：计算矩阵的特征值
	-numpy.linalg.eigvalsh(a, UPLO)：计算 Hermitian 或真实对称矩阵的特征值
	-numpy.linalg.norm(x ,ord,axis,keepdims)：计算矩阵或向量范数
	-numpy.linalg.cond(x ,p)：计算矩阵的条件数
	-numpy.linalg.det(a)：计算数组的行列式
	-numpy.linalg.matrix_rank(M ,tol)：使用奇异值分解方法返回秩
	-numpy.linalg.slogdet(a)：计算数组的行列式的符号和自然对数
	-numpy.trace(a ,offset,axis1,axis2,dtype,out)：沿数组的对角线返回总和-迹
	-numpy.linalg.solve(a,b)：求解线性矩阵方程或线性标量方程组
	-numpy.linalg.tensorsolve(a,b ,axes)：为 x 解出张量方程a x = b
	-numpy.linalg.lstsq(a,b ,rcond)：将最小二乘解返回到线性矩阵方程
	-numpy.linalg.inv(a)：计算逆矩阵
	-numpy.linalg.pinv(a ,rcond)：计算矩阵的（Moore-Penrose）伪逆
	-numpy.linalg.tensorinv(a ,ind)：计算N维数组的逆

	
# 常见操作

## 创建递增的数列(一维数组)<br>
	tmp = np.arange(15);  
## 给定一维数组转化为制定格式的二维数组<br>
	tmp = np.arange(15).reshape(5, 1);
## 获取随机值<br>
    b = np.random.rand(2, 3);
    b = np.random.randn(2, 3);
    b = np.random.standard_normal((2, 3));
    b = np.random.randint(low=0, high=2, size=(2, 3), dtype=np.int32);
    b = np.random.random_integers(low=0, high=2, size=(2, 3));
    b = np.random.randint(low=0, high=2, size=(2, 3), dtype=np.int32);
    b = np.random.random_integers(low=0, high=2, size=(2, 3));
    b = np.random.random_sample(size=(2, 3));
    np.random.shuffle(b);
## 求逆矩阵<br>
    mat = np.array([[1, 2], [3, 4]])
    det = np.linalg.inv(mat)
## 求矩阵行列式值<br>
    mat = np.array([[1, 2], [3, 4]])
    det = np.linalg.det(mat)
## 求矩阵行特征值&特征向量
    mat = np.array([[1, 2], [3, 4]])
    det1, det2 = np.linalg.eig(mat)
## 计算矩阵内积
    mat = np.array([[1, 2], [3, 4]])
    print(np.dot(mat.T, mat))
## 求转置矩阵
    y = np.transpose(x)
## 统计元素出现次数
    np.bincount(mat)
## 求加权平均值<br>
    np.average(mat, axis=1, weights=[2, 1])

## 矩阵按照第一列元素大小对整个矩阵进行行排序
    mat1=mat1[mat1[:,0].argsort()]
    
## 多维数组/矩阵降序排列
    方案1:
    list1 = [[1, 3, 2], [3, 1, 4]]
    a = numpy.array(list1)
    a = numpy.array([a[line_id,i] for line_id, i in enumerate(argsort(-a, axis=1))])
    
    方案2:
    list1 = [[1, 3, 2], [3, 1, 4]]
    a = numpy.array(list1)
    sindx = argsort(-a, axis=1)
    indx = numpy.meshgrid(*[numpy.arange(x) for x in a.shape], sparse=True,
                       indexing='ij')
    indx[1] = sindx
    a = a[indx]
    
    方案3：
    list1 = [[1, 3, 2], [3, 1, 4]]
    a = numpy.array(list1)
    a = -sort(-a, axis=1)
