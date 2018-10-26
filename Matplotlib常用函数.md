# Matplotlib常用函数 

# 主文

## matplotlib.figure() 
	功能：创建画布
	参数：num=序号或名称；dpi=分辨率；facecolor=背景颜色； edgecolor=边框颜色；figsize=图像大小
	返回值：matplotlib.figure.Figure
## matplotlib,plot()
	功能：创建数据点集
	参数：x, y, color="颜色", linewidth=线宽, linestyle="样式"
	返回值：list

    
 字符 | 颜色
---|---
| "b"  | 蓝色
| "g"  | 绿色
| "r"  | 红色
| "c"  | 青色
| "m"  | 品红
| "y"  | 黄色
| "k"  | 黑色
| "w"  | 白色
<br>

字符| 样式
---|---
| "-"  | 实线
| "--" | 虚线
| "-." | 点线
| ":" | 点虚线
| "." | 点
| "," | 像素
| "o" | 圆形
| "v" | 三角形
| "^" | 三角形
| "<" | 三角形
| ">" | 三角形
| "s" | 正方形
| "p" | 五角形
| "*" | 星形
| "h" | 1号六角形
| "H" | 2号六角形
| "+" | +号
| "x" | x号
| "D" | 钻石形
| "d" | 小钻石形
| "|" | 垂直线形
| "_" |  水平线形
<br>

	
## matplotlib.xlim()& matplotlib.ylim
	功能：设置坐标轴范围
	参数：极小值，极大值
	返回值：无
	
## matplotlib.xlabel()&matplotlib.ylabel()
	功能：设置坐标轴名称
	参数：名称，fontsize=字体大小，verticalalignment="垂直类型"，horizontalalignment="水平类型"，rotation="文字旋转类型"
	返回值：无
	
	verticalalignment参数可选：’top’, ‘bottom’, ‘center’, ‘baseline’ 
	horizontalalignment参数可选：’center’, ‘right’, ‘left’ 
	rotation参数可选:：‘vertical’,’horizontal’,’vertical’
    
## matplotlib.title()
	功能：设置画布标题
	参数：名称，fontsize=字体大小, fontstyle="文字样式", fontweight="文字粗细", alpha=透明度(0-1)，verticalalignment="垂直类型"，horizontalalignment="水平类型"，rotation="文字旋转类型"
	返回值：无
	
	fontstyle参数可选： 'normal'，'italic' ，'oblique'
	verticalalignment参数可选：’top’, ‘bottom’, ‘center’, ‘baseline’ 
	horizontalalignment参数可选：’center’, ‘right’, ‘left’ 
	rotation参数可选:：‘vertical’,’horizontal’,’vertical’
	
## matplotlib.xticks()
	功能：设置X轴标签
	参数：numpy.linspace(start, stop, number)
	返回值：无
	
## matplotlib.yticks()
	功能：设置Y轴标签
	参数：[-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$']]
	
## matplotlib.gca()
	功能：获取当前坐标轴信息
	参数：无
	返回值：matplotlib.axes._subplots.AxesSubplot
	
## matplot.legend()
	功能：设置图例信息
	参数：labels=["标签1", "标签2"], loc="位置", frameon=是否绘制边框, title="标题", edgecolor="边框颜色"
	返回：无
	
## matplotlib.scatter()
	功能：设置一系列散点
	参数：x，y，s="每个点的大小"，marker(标记类型)，color="颜色"
	返回：matplotlib.collections.PathCollection
	
## matplotlib.text()
	功能：添加文字
	参数：-0.5, 0.5, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
             fontdict={'size': 16, 'color': 'r'}
	返回：无
	
## matplotlib.annotate()
	功能：添加注解
	参数："local", xy=(0.5, 1), xytext=(0.5, 0.5), arrowprops=dict(facecolor='black', shrink=0.001)
	返回：无
	
## matplotlib.bar()
	功能：设置条状图
	参数：X，Y，width=宽度, align="对齐方式(center/edge)", orientation="方向(vertical/horizontal)"， facecolor='矩形颜色', edgecolor='边框颜色'
	返回：matplotlib.container.BarContainer
	
## matplotlib.meshgrid()
	功能：从一个坐标向量中返回一个目标矩阵
	参数：X(numpy.ndarray类型)，Y(numpy.ndarray类型)
	返回：numpy.ndarray，numpy.ndarray
	说明：https://blog.csdn.net/sinat_29957455/article/details/78825945
	
## matplotlib.imshow()
	功能：绘制图像
	参数：X(要绘制的图像或数组)，cmap=颜色图谱，interpolation=？？？，origin="lower"
	返回：matplotlib.image.AxesImage
	cmap参数值参考：http://www.cnblogs.com/denny402/p/5122594.html或者https://matplotlib.org/examples/color/colormaps_reference.html
	cmap格式：cmap=plt.cm.hot或者cmap=plt.get_cmap('hsv')
    函数解释参考：https://blog.csdn.net/goldxwang/article/details/76855200
	interpolation参数值参考：https://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
	主要用处：热图！
	
## matplotlib.colorbar()
	功能：添加颜色类标
	参数：shrink=透明度(0-1之间)
	返回：无
	
## Axes3D()
	功能：创建3D绘图模块
	参数：matplotlib.figure()
	返回：mpl_toolkits.mplot3d.axes3d.Axes3D
	
## Axes3D.plot_surface()
	功能：创建三维曲面
	参数：X，Y，Z，rstride=1(row的跨度), cstride=1(column的跨度), cmap=颜色图谱(同上)
	返回：无
	
## Axes3.contourf()
	功能：创建投影
	参数：(X, Y, Z, zdir='z', offset=-2, cmap=颜色图谱(plt.get_cmap('rainbow'))
	返回：无
	说明：zdir=x，代表对XZ平面投影
	
## matplotlib.subplot()
	功能：创建子图
	参数：x1，x2，x3(行，列，坐标)
	返回：matplotlib.axes._subplots.AxesSubplot
	index以及网格划分参考：https://blog.csdn.net/claroja/article/details/70841382

## matplotlib.figure().add_axes()
	功能：添加画面
	参数：[x,y,length,weight] (以左下角为基点的起始点，长宽，参数范围：0-1)
	返回：matplotlib.axes._axes.Axes
	
	
# 常见操作

## 设置某坐标轴不见
``` javascript
	frame = plt.gca()
	# y 轴不可见
	frame.axes.get_yaxis().set_visible(False)
	# x 轴不可见
	frame.axes.get_xaxis().set_visible(False)
	# 关闭所有轴
	plt.axis("off")
```

## 设置某图像的某条变不可见
``` javascript
	frame = plt.gca()
    frame.spines["top"].set_color("none")
	frame.spines["bottom"].set_color("none")
	frame.spines["right"].set_color("none")
	frame.spines["left"].set_color("none")
```

## 设置X轴/Y轴坐标刻度数字或名称的位置
``` javascript
	frame = plt.gca()
	frame.xaxis.set_ticks_position("bottom")
	frame.xaxis.set_ticks_position("top")
	frame.xaxis.set_ticks_position("both")
```
## 设置X轴/Y轴起点的绝对位置

``` javascript
	frame = plt.gca()
	frame.xaxis.set_ticks_position("bottom")
        frame.spines["bottom"].set_position(("data", 0))
        frame.yaxis.set_ticks_position("left")
        frame.spines["left"].set_position(("data", 0))
```

## 设置图例

``` javascript
	# 设置图例
	plt.legend(labels=["up", "down"], loc="best", frameon=False, title="afs", edgecolor="blue")
	# 清除图例
	ax1.legend_.remove() ##移除子图ax1中的图例
	# 分别绘制图例
	line1, = plt.plot([1,2,3], label="Line 1", linestyle='--')  
	line2, = plt.plot([3,2,1], label="Line 2", linewidth=4)  
	first_legend = plt.legend(handles=[line1], loc=1)  
	ax = plt.gca().add_artist(first_legend)  
	plt.legend(handles=[line2], loc=4) 
```

## 给定一个点做出其到X轴的垂线，并对该点进行标记
``` javascript
	x0 = 1
	y0 = 2*x0 + 1
	plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
	plt.scatter([x0, ], [y0, ], s=50, color='b')
```

## 添加文字
``` javascript
	 plt.text(-0.5, 0.5, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})
```

## 添加注解
``` javascript
	plt.annotate("local", xy=(0.5, 1), xytext=(0.5, 0.5), arrowprops=dict(facecolor='black', shrink=0.001))
```

## 绘制直方图并附带文字
``` javascript
	n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    for x, y in zip(X, Y1):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x + 0.0, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    for x, y in zip(X, Y2):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x + 0.0, -y - 0.05, '%.2f' % y, ha='center', va='top')

    plt.xlim(-.5, n)
    # plt.xticks(())
    plt.ylim(-1.25, 1.25)
    # plt.yticks(())

    plt.show()
```

## 绘制等高线
``` javascript
	n = 256
    x = np.linspace(-3, 3, n)
    # print(type(x))
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    # print(type(X))

    # use plt.contourf to filling contours
    # X, Y and value for (X,Y) point
    plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    # use plt.contour to add contour lines
    C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
    # adding label
    plt.clabel(C, inline=True, fontsize=10)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()
```

## 绘制像素图
``` javascript
	# image data
    a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
                  0.365348418405, 0.439599930621, 0.525083754405,
                  0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)

    a = plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
    print(type(a))
    plt.colorbar(shrink=.92)
    # plt.xticks(())
    # plt.yticks(())
    plt.show()
```

## 绘制3D图像
	[演示参考1](https://blog.csdn.net/dahunihao/article/details/77833877)
	[演示参考2](https://blog.csdn.net/shu15121856/article/details/72590620)
	
``` javascript
	fig = plt.figure()
    ax = Axes3D(fig)
    print(type(ax))
    # X, Y value
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    # height value
    print(R)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('hsv'))
    =============================================================
            Argument      Description
            *X*, *Y*, *Z* Data values as 2D arrays
            *rstride*     Array row stride (step size), defaults to 10
            *cstride*     Array column stride (step size), defaults to 10
            *color*       Color of the surface patches
            *cmap*        A colormap for the surface patches.
            *facecolors*  Face colors for the individual patches
            *norm*        An instance of Normalize to map values to colors
            *vmin*        Minimum value to map
            *vmax*        Maximum value to map
            *shade*       Whether to shade the facecolors
	 =============================================================
    # I think this is different from plt12_contours
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('hsv'))

     =============================================================
            Argument    Description          
            *X*, *Y*,   Data values as numpy.arrays
            *Z*
            *zdir*      The direction to use: x, y or z (default)
            *offset*    If specified plot a projection of the filled contour
                        on this position in plane normal to zdir
      =============================================================
    ax.set_zlim(-2, 2)
    plt.show()
```  

## 绘制子网格
index以及网格划分参考：https://blog.csdn.net/claroja/article/details/70841382
``` javascript
	plt.figure(1)  # 创建第一个画板（figure）
    apx = plt.subplot(211)  # 第一个画板的第一个子图
    print(type(apx))
    plt.plot([1, 2, 3])
    plt.subplot(212)  # 第二个画板的第二个子图
    plt.plot([4, 5, 6])
    plt.show()
```

## 绘制多重子网格
``` javascript
	plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)  # stands for axes
    ax1.plot([1, 2], [1, 2])
    ax1.set_title('ax1_title')
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax4.scatter([1, 2], [2, 2])
    ax4.set_xlabel('ax4_x')
    ax4.set_ylabel('ax4_y')
    ax5 = plt.subplot2grid((3, 3), (2, 1))
    plt.show()
```
## 画中画
``` javascript
	fig = plt.figure()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 3, 4, 2, 5, 8, 6]
    # below are all percentage
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])  # main axes
    ax1.plot(x, y, 'r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('title')

    ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
    ax2.plot(y, x, 'b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('title inside 1')

    # different method to add axes
    ####################################
    plt.axes([0.6, 0.2, 0.25, 0.25])
    plt.plot(y[::-1], x, 'g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('title inside 2')

    plt.show()
```
## 绘制饼图
``` javascript
	# 设置标签
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    # 各个标签所占比例
    sizes = [15, 30, 45, 10]
    # 是否分离or分离距离
    explode = (0, 0.25, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
	# 参数：data、？、标签、保留小数点多少位、阴影、起始角度
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
```
## 绘制流向图
``` javascript
	w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    U = -1 - X ** 2 + Y
    V = 1 + X - Y ** 2
    speed = np.sqrt(U * U + V * V)

    fig = plt.figure(figsize=(7, 9))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Varying Density')

    # Varying color along a streamline
    ax1 = fig.add_subplot(gs[0, 1])
    strm = ax1.streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')
    fig.colorbar(strm.lines)
    ax1.set_title('Varying Color')

    #  Varying line width along a streamline
    ax2 = fig.add_subplot(gs[1, 0])
    lw = 5 * speed / speed.max()
    ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)
    ax2.set_title('Varying Line Width')

    # Controlling the starting points of the streamlines
    seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1, 0, 1, 2, 2]])

    ax3 = fig.add_subplot(gs[1, 1])
    strm = ax3.streamplot(X, Y, U, V, color=U, linewidth=2,
                          cmap='autumn', start_points=seed_points.T)
    fig.colorbar(strm.lines)
    ax3.set_title('Controlling Starting Points')

    # Displaying the starting points with blue symbols.
    ax3.plot(seed_points[0], seed_points[1], 'bo')
    ax3.axis((-w, w, -w, w))

    # Create a mask
    mask = np.zeros(U.shape, dtype=bool)
    mask[40:60, 40:60] = True
    U[:20, :20] = np.nan
    U = np.ma.array(U, mask=mask)

    ax4 = fig.add_subplot(gs[2:, :])
    ax4.streamplot(X, Y, U, V, color='r')
    ax4.set_title('Streamplot with Masking')

    ax4.imshow(~mask, extent=(-w, w, -w, w), alpha=0.5,
               interpolation='nearest', cmap='gray', aspect='auto')
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.show()
```
## 高级散点图
``` javascript
	# Load a numpy record array from yahoo csv data with fields date, open, close,
	# volume, adj_close from the mpl-data/example directory. The record array
	# stores the date as an np.datetime64 with a day unit ('D') in the date column.
	with cbook.get_sample_data('goog.npz') as datafile:
		price_data = np.load(datafile)['price_data'].view(np.recarray)
	price_data = price_data[-250:]  # get the most recent 250 trading days

	delta1 = np.diff(price_data.adj_close) / price_data.adj_close[:-1]

	# Marker size in units of points^2
	volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
	close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

	fig, ax = plt.subplots()
	ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

	ax.set_xlabel(r'$\Delta_i$', fontsize=15)
	ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
	ax.set_title('Volume and percent change')

	ax.grid(True)
	fig.tight_layout()

	plt.show()
```

## 表格
``` javascript
	data = [[ 66386, 174296,  75131, 577908,  32015],
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]

	columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
	rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

	values = np.arange(0, 2500, 500)
	value_increment = 1000

	# Get some pastel shades for the colors
	colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
	n_rows = len(data)

	index = np.arange(len(columns)) + 0.3
	bar_width = 0.4

	# Initialize the vertical-offset for the stacked bar chart.
	y_offset = np.zeros(len(columns))

	# Plot bars and create text labels for the table
	cell_text = []
	for row in range(n_rows):
		plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
		y_offset = y_offset + data[row]
		cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
	# Reverse colors and text labels to display the last value at the top.
	colors = colors[::-1]
	cell_text.reverse()

	# Add a table at the bottom of the axes
	the_table = plt.table(cellText=cell_text,
						  rowLabels=rows,
						  rowColours=colors,
						  colLabels=columns,
						  loc='bottom')

	# Adjust layout to make room for the table:
	plt.subplots_adjust(left=0.2, bottom=0.2)

	plt.ylabel("Loss in ${0}'s".format(value_increment))
	plt.yticks(values * value_increment, ['%d' % val for val in values])
	plt.xticks([])
	plt.title('Loss by Disaster')

	plt.show()
```
## 日期标记标签

``` javascript
	years = mdates.YearLocator()   # every year
	months = mdates.MonthLocator()  # every month
	yearsFmt = mdates.DateFormatter('%Y')

	# Load a numpy record array from yahoo csv data with fields date, open, close,
	# volume, adj_close from the mpl-data/example directory. The record array
	# stores the date as an np.datetime64 with a day unit ('D') in the date column.
	with cbook.get_sample_data('goog.npz') as datafile:
		r = np.load(datafile)['price_data'].view(np.recarray)

	fig, ax = plt.subplots()
	ax.plot(r.date, r.adj_close)

	# format the ticks
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(yearsFmt)
	ax.xaxis.set_minor_locator(months)

	# round to nearest years...
	datemin = np.datetime64(r.date[0], 'Y')
	datemax = np.datetime64(r.date[-1], 'Y') + np.timedelta64(1, 'Y')
	ax.set_xlim(datemin, datemax)


	# format the coords message box
	def price(x):
		return '$%1.2f' % x
	ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
	ax.format_ydata = price
	ax.grid(True)

	# rotates and right aligns the x labels, and moves the bottom of the
	# axes up to make room for them
	fig.autofmt_xdate()

	plt.show()
```

